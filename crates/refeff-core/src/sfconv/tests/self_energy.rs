use super::*;

#[test]
fn senergies_beta_helpers_match_feff_reference() -> Result<(), SfconvError> {
    let lowq0_context = senergies_reference_context(false);
    assert_close(
        sfconv_free_electron_exchange(1.0, lowq0_context.fermi_momentum)?,
        -std::f64::consts::FRAC_1_PI,
        1.0e-15,
    );
    assert_close(
        sfconv_free_electron_exchange(1.35, lowq0_context.fermi_momentum)?,
        -0.133_662_411_513_184_28,
        1.0e-15,
    );
    assert_close(
        sfconv_extrinsic_beta(0.36, lowq0_context)?,
        0.287_008_463_933_952_74,
        1.0e-14,
    );
    assert_close(
        sfconv_extrinsic_beta(0.95, lowq0_context)?,
        0.099_242_494_271_372_31,
        1.0e-14,
    );
    assert_close(
        sfconv_imaginary_self_energy(0.36, lowq0_context)?,
        -0.901_663_681_812_997,
        1.0e-14,
    );

    let lowq1_context = senergies_reference_context(true);
    assert_close(sfconv_extrinsic_beta(-0.20, lowq1_context)?, 0.0, 0.0);
    assert_close(
        sfconv_extrinsic_beta(0.36, lowq1_context)?,
        0.287_008_463_933_952_74,
        1.0e-14,
    );
    assert_close(
        sfconv_imaginary_self_energy(-0.20, lowq1_context)?,
        0.0,
        0.0,
    );
    Ok(())
}

#[test]
fn senergies_real_self_energy_matches_feff_reference() -> Result<(), SfconvError> {
    let pkgt_lowq0_context = senergies_reference_context(false);
    assert_close(
        sfconv_real_self_energy_integrand_upper(0.55, 0.36, pkgt_lowq0_context)?,
        2.874_639_111_469_788_7,
        1.0e-14,
    );
    assert_close(
        sfconv_real_self_energy_integrand_middle(0.55, 0.36, pkgt_lowq0_context)?,
        5.222_817_359_927_24,
        1.0e-14,
    );
    assert_close(
        sfconv_real_self_energy_integrand_lower(0.55, 0.36, pkgt_lowq0_context)?,
        -8.010_746_486_392_092,
        1.0e-14,
    );
    let real_pkgt = sfconv_real_self_energy(0.36, pkgt_lowq0_context)?;
    assert_close(real_pkgt.value, -0.707_783_970_737_988_9, 1.0e-12);
    assert!(real_pkgt.evaluations > 0);
    assert!(real_pkgt.max_regions > 0);
    assert_close(
        sfconv_real_self_energy(0.95, pkgt_lowq0_context)?.value,
        0.196_748_431_942_598_25,
        1.0e-12,
    );

    let pkgt_lowq1_context = senergies_reference_context(true);
    assert_close(
        sfconv_real_self_energy(-0.20, pkgt_lowq1_context)?.value,
        -0.277_039_230_882_649,
        1.0e-12,
    );

    let pklt_lowq0_context = SfconvSelfEnergyContext {
        photoelectron_momentum: 0.82,
        ..senergies_reference_context(false)
    };
    assert_close(
        sfconv_real_self_energy_integrand_upper(0.55, 0.36, pklt_lowq0_context)?,
        -3.190_158_193_028_965_5,
        1.0e-14,
    );
    assert_close(
        sfconv_real_self_energy_integrand_middle(0.55, 0.36, pklt_lowq0_context)?,
        0.649_162_805_914_428_2,
        1.0e-14,
    );
    assert_close(
        sfconv_real_self_energy_integrand_lower(0.55, 0.36, pklt_lowq0_context)?,
        -2.194_055_192_971_564_6,
        1.0e-14,
    );
    assert_close(
        sfconv_real_self_energy(0.36, pklt_lowq0_context)?.value,
        -0.077_377_126_607_744_2,
        1.0e-12,
    );

    let pklt_lowq1_context = SfconvSelfEnergyContext {
        include_below_fermi: true,
        ..pklt_lowq0_context
    };
    assert_close(
        sfconv_real_self_energy_integrand_middle(0.55, 0.36, pklt_lowq1_context)?,
        -0.291_337_926_232_215_6,
        1.0e-14,
    );
    assert_close(
        sfconv_real_self_energy(0.36, pklt_lowq1_context)?.value,
        0.021_796_867_569_840_478,
        1.0e-12,
    );

    let pkeq_context = SfconvSelfEnergyContext {
        photoelectron_momentum: 1.0,
        ..senergies_reference_context(false)
    };
    assert_close(
        sfconv_real_self_energy(0.36, pkeq_context)?.value,
        0.043_101_938_251_358_85,
        1.0e-12,
    );
    Ok(())
}

#[test]
fn senergies_self_energy_derivatives_match_feff_reference() -> Result<(), SfconvError> {
    let pkgt_lowq0_context = senergies_reference_context(false);
    assert_close(
        sfconv_real_self_energy_derivative_integrand_upper(0.55, 0.36, pkgt_lowq0_context)?,
        10.732_547_867_812_46,
        1.0e-13,
    );
    assert_close(
        sfconv_real_self_energy_derivative_integrand_middle(0.55, 0.36, pkgt_lowq0_context)?,
        18.720_823_012_355_86,
        1.0e-13,
    );
    assert_close(
        sfconv_real_self_energy_derivative_integrand_lower(0.55, 0.36, pkgt_lowq0_context)?,
        -20.398_432_856_236_53,
        1.0e-13,
    );
    let real_derivative = sfconv_real_self_energy_derivative(0.36, pkgt_lowq0_context)?;
    assert_close(real_derivative.value, 2.961_445_535_932_464, 1.0e-12);
    assert!(real_derivative.evaluations > 0);
    assert!(real_derivative.max_regions > 0);
    assert_close(
        sfconv_real_self_energy_derivative(0.95, pkgt_lowq0_context)?.value,
        -0.034_316_545_918_129_96,
        1.0e-12,
    );
    assert_close(
        sfconv_imaginary_self_energy_derivative(0.36, pkgt_lowq0_context)?,
        -6.610_090_947_687_186,
        1.0e-12,
    );
    assert_close(
        sfconv_imaginary_self_energy_derivative(0.95, pkgt_lowq0_context)?,
        0.400_030_527_250_079_1,
        1.0e-12,
    );

    let pkgt_lowq1_context = senergies_reference_context(true);
    assert_close(
        sfconv_real_self_energy_derivative(-0.20, pkgt_lowq1_context)?.value,
        -0.452_613_488_967_939_7,
        1.0e-12,
    );
    assert_close(
        sfconv_imaginary_self_energy_derivative(-0.20, pkgt_lowq1_context)?,
        0.0,
        0.0,
    );
    assert_close(
        sfconv_real_self_energy_derivative_integrand_middle(0.55, 0.36, pkgt_lowq1_context)?,
        18.394_386_251_356_508,
        1.0e-13,
    );
    assert_close(
        sfconv_real_self_energy_derivative(0.36, pkgt_lowq1_context)?.value,
        2.951_013_422_721_360_7,
        1.0e-12,
    );
    assert_close(
        sfconv_imaginary_self_energy_derivative(0.36, pkgt_lowq1_context)?,
        -6.610_090_947_687_186,
        1.0e-12,
    );

    let pklt_lowq0_context = SfconvSelfEnergyContext {
        photoelectron_momentum: 0.82,
        ..senergies_reference_context(false)
    };
    assert_close(
        sfconv_real_self_energy_derivative_integrand_upper(0.55, 0.36, pklt_lowq0_context)?,
        17.650_634_174_439_2,
        1.0e-13,
    );
    assert_close(
        sfconv_real_self_energy_derivative_integrand_middle(0.55, 0.36, pklt_lowq0_context)?,
        28.479_960_013_795_644,
        1.0e-13,
    );
    assert_close(
        sfconv_real_self_energy_derivative_integrand_lower(0.55, 0.36, pklt_lowq0_context)?,
        -0.329_585_793_363_548_93,
        1.0e-13,
    );
    assert_close(
        sfconv_real_self_energy_derivative(0.36, pklt_lowq0_context)?.value,
        0.540_035_967_831_518_6,
        1.0e-12,
    );
    assert_close(
        sfconv_imaginary_self_energy_derivative(0.36, pklt_lowq0_context)?,
        0.295_448_827_556_208_1,
        1.0e-12,
    );

    let pklt_lowq1_context = SfconvSelfEnergyContext {
        include_below_fermi: true,
        ..pklt_lowq0_context
    };
    assert_close(
        sfconv_real_self_energy_derivative_integrand_middle(0.55, 0.36, pklt_lowq1_context)?,
        27.874_936_402_523_584,
        1.0e-13,
    );
    assert_close(
        sfconv_real_self_energy_derivative(0.36, pklt_lowq1_context)?.value,
        0.664_935_465_444_516_1,
        1.0e-12,
    );
    assert_close(
        sfconv_imaginary_self_energy_derivative(0.36, pklt_lowq1_context)?,
        0.295_448_827_556_208_1,
        1.0e-12,
    );

    let pkeq_context = SfconvSelfEnergyContext {
        photoelectron_momentum: 1.0,
        ..senergies_reference_context(false)
    };
    assert_close(
        sfconv_real_self_energy_derivative(0.36, pkeq_context)?.value,
        0.468_906_060_872_854_14,
        1.0e-12,
    );
    assert_close(
        sfconv_imaginary_self_energy_derivative(0.36, pkeq_context)?,
        0.750_887_782_735_307_7,
        1.0e-12,
    );
    Ok(())
}
