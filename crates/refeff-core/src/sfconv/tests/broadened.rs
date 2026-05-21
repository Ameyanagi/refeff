use super::*;

#[test]
fn brsigma_broadened_integrands_match_feff_formulas() -> Result<(), SfconvError> {
    let input = SfconvBroadenedSelfEnergyIntegrandInput {
        momentum: 0.73,
        energy: 0.21,
        context: senergies_reference_context(false),
    };
    let expected = [
        (
            SfconvBroadenedSelfEnergyBranch::ParticlePair,
            SfconvBroadenedSelfEnergyIntegrands {
                log_real: -7.011_705_793_259_941,
                log_imag: -0.369_504_267_018_922_97,
                atan_real: 0.325_185_453_673_107_86,
                atan_imag: 6.170_712_852_111_19,
            },
        ),
        (
            SfconvBroadenedSelfEnergyBranch::ParticleFermi,
            SfconvBroadenedSelfEnergyIntegrands {
                log_real: -13.797_429_315_272_487,
                log_imag: -0.727_099_675_343_745_2,
                atan_real: 0.070_287_942_851_676_6,
                atan_imag: 1.333_782_638_196_666_4,
            },
        ),
        (
            SfconvBroadenedSelfEnergyBranch::HoleFermi,
            SfconvBroadenedSelfEnergyIntegrands {
                log_real: 0.953_851_591_976_764_5,
                log_imag: 0.050_266_260_982_741_846,
                atan_real: 0.000_611_333_609_869_711_8,
                atan_imag: 0.011_600_654_705_615_22,
            },
        ),
        (
            SfconvBroadenedSelfEnergyBranch::HolePair,
            SfconvBroadenedSelfEnergyIntegrands {
                log_real: 7.129_828_634_229_525,
                log_imag: 0.375_729_127_995_351,
                atan_real: 0.324_874_938_869_935_85,
                atan_imag: 6.164_820_529_238_007,
            },
        ),
    ];

    for (branch, expected_integrands) in expected {
        let actual = sfconv_broadened_self_energy_integrands(branch, input)?;
        assert_close(actual.log_real, expected_integrands.log_real, 1.0e-13);
        assert_close(actual.log_imag, expected_integrands.log_imag, 1.0e-13);
        assert_close(actual.atan_real, expected_integrands.atan_real, 1.0e-13);
        assert_close(actual.atan_imag, expected_integrands.atan_imag, 1.0e-13);
    }
    Ok(())
}

#[test]
fn dbrsigma_broadened_derivative_integrands_match_feff_formulas() -> Result<(), SfconvError> {
    let input = SfconvBroadenedSelfEnergyIntegrandInput {
        momentum: 0.73,
        energy: 0.21,
        context: senergies_reference_context(false),
    };
    let expected = [
        (
            SfconvBroadenedSelfEnergyBranch::ParticlePair,
            SfconvBroadenedSelfEnergyDerivativeIntegrands {
                log_real: 8.237_536_803_919_268,
                log_imag: 0.434_103_353_523_399_8,
                atan_real: 0.042_642_154_194_309_74,
                atan_imag: 0.809_176_689_659_211_2,
            },
        ),
        (
            SfconvBroadenedSelfEnergyBranch::ParticleFermi,
            SfconvBroadenedSelfEnergyDerivativeIntegrands {
                log_real: -27.330_804_124_143_9,
                log_imag: -1.440_284_153_769_992_4,
                atan_real: 1.193_324_638_256_228,
                atan_imag: 22.644_505_154_990_576,
            },
        ),
        (
            SfconvBroadenedSelfEnergyBranch::HoleFermi,
            SfconvBroadenedSelfEnergyDerivativeIntegrands {
                log_real: -0.331_054_683_886_457,
                log_imag: -0.017_445_985_601_711,
                atan_real: -0.000_853_016_831_865_627_2,
                atan_imag: -0.016_186_830_831_466_846,
            },
        ),
        (
            SfconvBroadenedSelfEnergyBranch::HolePair,
            SfconvBroadenedSelfEnergyDerivativeIntegrands {
                log_real: 8.400_819_219_599_3,
                log_imag: 0.442_708_042_753_362_3,
                atan_real: -0.044_853_381_319_184_02,
                atan_imag: -0.851_136_892_627_315_1,
            },
        ),
    ];

    for (branch, expected_integrands) in expected {
        let actual = sfconv_broadened_self_energy_derivative_integrands(branch, input)?;
        assert_close(actual.log_real, expected_integrands.log_real, 1.0e-13);
        assert_close(actual.log_imag, expected_integrands.log_imag, 1.0e-13);
        assert_close(actual.atan_real, expected_integrands.atan_real, 1.0e-13);
        assert_close(actual.atan_imag, expected_integrands.atan_imag, 1.0e-13);
    }
    Ok(())
}

#[test]
fn brsigma_broadened_integrands_reject_invalid_inputs() {
    let input = SfconvBroadenedSelfEnergyIntegrandInput {
        momentum: -0.10,
        energy: 0.21,
        context: senergies_reference_context(false),
    };
    assert_eq!(
        sfconv_broadened_self_energy_integrands(
            SfconvBroadenedSelfEnergyBranch::ParticlePair,
            input,
        ),
        Err(SfconvError::InvalidIntegrationInterval {
            lower: -0.10,
            upper: 0.0,
        })
    );
    assert_eq!(
        sfconv_broadened_self_energy_derivative_integrands(
            SfconvBroadenedSelfEnergyBranch::ParticlePair,
            input,
        ),
        Err(SfconvError::InvalidIntegrationInterval {
            lower: -0.10,
            upper: 0.0,
        })
    );

    let zero_broadening = SfconvBroadenedSelfEnergyIntegrandInput {
        momentum: 0.73,
        energy: 0.21,
        context: SfconvSelfEnergyContext {
            pole_broadening: 0.0,
            ..senergies_reference_context(false)
        },
    };
    assert_eq!(
        sfconv_broadened_self_energy_integrands(
            SfconvBroadenedSelfEnergyBranch::ParticlePair,
            zero_broadening,
        ),
        Err(SfconvError::NonPositiveScalar {
            field: "pole_broadening",
            value: 0.0,
        })
    );
    assert_eq!(
        sfconv_broadened_self_energy_derivative_integrands(
            SfconvBroadenedSelfEnergyBranch::ParticlePair,
            zero_broadening,
        ),
        Err(SfconvError::NonPositiveScalar {
            field: "pole_broadening",
            value: 0.0,
        })
    );
}

#[test]
fn brsigma_broadened_self_energy_matches_feff_reference() -> Result<(), SfconvError> {
    let cases = [
        (
            0.36,
            senergies_reference_context(false),
            -0.518_548_796_704_916_7,
            -0.820_845_165_208_279_3,
        ),
        (
            -0.20,
            senergies_reference_context(true),
            -0.276_438_440_404_569,
            -0.012_356_840_692_487_325,
        ),
        (
            0.36,
            SfconvSelfEnergyContext {
                photoelectron_momentum: 0.82,
                ..senergies_reference_context(false)
            },
            -0.090_781_303_269_171_75,
            -0.280_887_927_239_661_94,
        ),
        (
            0.36,
            SfconvSelfEnergyContext {
                photoelectron_momentum: 0.82,
                include_below_fermi: true,
                ..senergies_reference_context(false)
            },
            0.008_365_301_760_209_81,
            -0.284_132_323_784_229_7,
        ),
        (
            0.36,
            SfconvSelfEnergyContext {
                photoelectron_momentum: 1.0,
                ..senergies_reference_context(false)
            },
            0.013_728_093_655_548_983,
            -0.412_629_377_510_605_5,
        ),
    ];

    for (energy, context, expected_real, expected_imaginary) in cases {
        let actual = sfconv_broadened_self_energy(energy, context)?;
        assert_close(actual.real, expected_real, 1.0e-12);
        assert_close(actual.imaginary, expected_imaginary, 1.0e-12);
        assert!(actual.real_estimated_error >= 0.0);
        assert!(actual.real_estimated_error < 1.0e-6);
        assert!(actual.imaginary_estimated_error >= 0.0);
        assert!(actual.imaginary_estimated_error < 1.0e-6);
        assert!(actual.evaluations > 0);
        assert!(actual.max_regions > 0);
    }
    Ok(())
}

#[test]
fn brsigma_broadened_self_energy_rejects_invalid_inputs() {
    let context = senergies_reference_context(false);
    assert!(matches!(
        sfconv_broadened_self_energy(f64::NAN, context),
        Err(SfconvError::NonFiniteScalar {
            field: "self-energy energy",
            ..
        })
    ));
    assert_eq!(
        sfconv_broadened_self_energy(
            0.36,
            SfconvSelfEnergyContext {
                pole_broadening: 0.0,
                ..context
            },
        ),
        Err(SfconvError::NonPositiveScalar {
            field: "pole_broadening",
            value: 0.0,
        })
    );
}

#[test]
fn dbrsigma_broadened_self_energy_derivative_matches_feff_reference() -> Result<(), SfconvError> {
    let cases = [
        (
            0.36,
            senergies_reference_context(false),
            2.953_632_555_240_584,
            -4.153_776_392_437_791,
        ),
        (
            -0.20,
            senergies_reference_context(true),
            -0.453_145_835_952_415_03,
            -0.046_313_231_462_640_74,
        ),
        (
            0.36,
            SfconvSelfEnergyContext {
                photoelectron_momentum: 0.82,
                ..senergies_reference_context(false)
            },
            0.533_248_980_604_782_1,
            0.196_090_288_785_958_72,
        ),
        (
            0.36,
            SfconvSelfEnergyContext {
                photoelectron_momentum: 0.82,
                include_below_fermi: true,
                ..senergies_reference_context(false)
            },
            0.467_087_536_743_928_44,
            0.199_768_325_815_296_63,
        ),
        (
            0.36,
            SfconvSelfEnergyContext {
                photoelectron_momentum: 1.0,
                ..senergies_reference_context(false)
            },
            0.462_197_179_911_945_2,
            0.647_423_140_545_274,
        ),
    ];

    for (energy, context, expected_real, expected_imaginary) in cases {
        let actual = sfconv_broadened_self_energy_derivative(energy, context)?;
        assert_close(actual.real, expected_real, 1.0e-12);
        assert_close(actual.imaginary, expected_imaginary, 1.0e-12);
        assert!(actual.real_estimated_error >= 0.0);
        assert!(actual.real_estimated_error < 1.0e-6);
        assert!(actual.imaginary_estimated_error >= 0.0);
        assert!(actual.imaginary_estimated_error < 1.0e-6);
        assert!(actual.evaluations > 0);
        assert!(actual.max_regions > 0);
    }
    Ok(())
}

#[test]
fn dbrsigma_broadened_self_energy_derivative_rejects_invalid_inputs() {
    let context = senergies_reference_context(false);
    assert!(matches!(
        sfconv_broadened_self_energy_derivative(f64::NAN, context),
        Err(SfconvError::NonFiniteScalar {
            field: "self-energy energy",
            ..
        })
    ));
    assert_eq!(
        sfconv_broadened_self_energy_derivative(
            0.36,
            SfconvSelfEnergyContext {
                pole_broadening: 0.0,
                ..context
            },
        ),
        Err(SfconvError::NonPositiveScalar {
            field: "pole_broadening",
            value: 0.0,
        })
    );
}
