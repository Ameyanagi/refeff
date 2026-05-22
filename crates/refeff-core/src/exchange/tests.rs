use super::*;

fn assert_real_close(actual: Real, expected: Real) {
    assert!(
        (actual - expected).abs() < 1.0e-14,
        "actual={actual}, expected={expected}, diff={}",
        (actual - expected).abs()
    );
}

#[test]
fn dirac_hara_exchange_potential_matches_feff_reference() -> Result<(), ExchangeError> {
    assert_real_close(
        dirac_hara_exchange_potential(2.0, 1.3)?,
        -0.127_197_685_551_790_5,
    );
    assert_eq!(dirac_hara_exchange_potential(150.0, 0.4)?, 0.0);
    Ok(())
}

#[test]
fn hedin_lundqvist_ffq_matches_feff_reference() -> Result<(), ExchangeError> {
    assert_real_close(
        hedin_lundqvist_ffq(0.8, 0.42, 1.2, 0.7, 4.0 / 3.0)?,
        0.086_644_481_840_666_34,
    );
    assert_real_close(
        hedin_lundqvist_ffq(1.6, 1.8, 2.4, 0.35, 0.9)?,
        0.062_528_814_886_981_41,
    );
    Ok(())
}

#[test]
fn von_barth_hedin_potential_matches_feff_reference() -> Result<(), ExchangeError> {
    assert_real_close(
        von_barth_hedin_potential(2.5, 1.2)?,
        -0.318_654_527_096_978_5,
    );
    assert_eq!(von_barth_hedin_potential(1200.0, 0.8)?, 0.0);
    Ok(())
}

#[test]
fn perdew_zunger_vxc_matches_feff_reference() -> Result<(), ExchangeError> {
    assert_real_close(perdew_zunger_vxc(0.75)?, -0.888_417_338_140_178);
    assert_real_close(perdew_zunger_vxc(2.0)?, -0.357_256_470_778_624_55);
    assert_real_close(perdew_zunger_vxc(10.0)?, -0.083_694_351_354_168_7);

    let full = perdew_zunger_exchange_correlation(2.0)?;
    assert_real_close(full.potential, -0.357_256_470_778_624_55);
    Ok(())
}

#[test]
fn perrot_dharma_wardana_vxc_matches_feff_reference() -> Result<(), ExchangeError> {
    assert_real_close(
        perrot_dharma_wardana_vxc(2.0, 0.0)?,
        -0.365_286_030_418_622_73,
    );
    assert_real_close(
        perrot_dharma_wardana_vxc(2.0, 0.05)?,
        -0.372_727_169_113_195_8,
    );
    assert_real_close(
        perrot_dharma_wardana_vxc(0.75, 0.12)?,
        -0.898_713_301_179_314_3,
    );
    assert_real_close(
        perrot_dharma_wardana_reduced_vxc(2.0, 0.5)?,
        -0.371_754_474_403_717,
    );
    assert_real_close(
        perrot_dharma_wardana_reduced_vxc(8.0, 4.0)?,
        -0.094_263_857_322_040_14,
    );
    Ok(())
}

#[test]
fn karasiev_sjostrom_dufty_trickey_matches_feff_reference() -> Result<(), ExchangeError> {
    assert_real_close(
        karasiev_sjostrom_dufty_trickey_vxc(2.0, 0.0)?,
        -0.356_646_002_509_705_63,
    );
    assert_real_close(
        karasiev_sjostrom_dufty_trickey_vxc(2.0, 0.05)?,
        -0.359_324_656_674_432_86,
    );
    assert_real_close(
        karasiev_sjostrom_dufty_trickey_vxc(0.75, 0.12)?,
        -0.887_463_405_178_325_2,
    );

    let zero = karasiev_sjostrom_dufty_trickey_free_energy(2.0, 0.0, KsdTSpin::Unpolarized)?;
    assert_real_close(zero.free_energy_per_particle, -0.273_783_738_220_404_75);
    assert_real_close(zero.potential, -0.356_646_002_509_705_63);

    let first = karasiev_sjostrom_dufty_trickey_free_energy(2.0, 0.5, KsdTSpin::Unpolarized)?;
    assert_real_close(first.free_energy_per_particle, -0.258_509_037_442_536_34);
    assert_real_close(first.potential, -0.351_894_120_250_991_87);
    assert_real_close(
        karasiev_sjostrom_dufty_trickey_internal_energy(2.0, 0.5, KsdTSpin::Unpolarized)?,
        -0.287_859_963_317_935_5,
    );

    let second = karasiev_sjostrom_dufty_trickey_free_energy(8.0, 4.0, KsdTSpin::Unpolarized)?;
    assert_real_close(second.free_energy_per_particle, -0.055_636_903_924_648_526);
    assert_real_close(second.potential, -0.080_007_059_893_872_29);
    assert_real_close(
        karasiev_sjostrom_dufty_trickey_internal_energy(8.0, 4.0, KsdTSpin::Unpolarized)?,
        -0.071_401_338_006_861_6,
    );

    let polarized =
        karasiev_sjostrom_dufty_trickey_free_energy(2.0, 0.5, KsdTSpin::FullyPolarized)?;
    assert_real_close(polarized.free_energy_per_particle, -0.271_613_018_966_040_1);
    assert_real_close(polarized.potential, -0.386_717_363_845_231_3);
    assert_real_close(
        karasiev_sjostrom_dufty_trickey_internal_energy(2.0, 0.5, KsdTSpin::FullyPolarized)?,
        -0.321_359_319_863_624_4,
    );
    Ok(())
}

#[test]
fn quinn_imaginary_self_energy_matches_feff_reference() -> Result<(), ExchangeError> {
    assert_real_close(
        quinn_imaginary_self_energy(1.15, 2.0, 0.65, 0.42)?,
        -0.002_466_676_087_856_595,
    );
    assert_real_close(
        quinn_imaginary_self_energy(2.4, 4.0, 0.35, 0.18)?,
        -0.000_005_424_606_390_748_76,
    );
    Ok(())
}

#[test]
fn hedin_lundqvist_imaginary_self_energy_matches_feff_reference() -> Result<(), ExchangeError> {
    let first = hedin_lundqvist_imaginary_self_energy(2.0, 1.3)?;
    assert_real_close(first.value, -0.014_367_116_812_766_725);
    assert!(!first.cusp);

    let second = hedin_lundqvist_imaginary_self_energy(5.0, 0.8)?;
    assert_real_close(second.value, -0.062_054_667_501_585_545);
    assert!(second.cusp);
    Ok(())
}

#[test]
fn hedin_lundqvist_self_energy_matches_feff_reference() -> Result<(), ExchangeError> {
    let first = hedin_lundqvist_self_energy(2.0, 1.3)?;
    assert_real_close(first.real, -0.368_652_535_973_926_2);
    assert_real_close(first.imaginary, -0.014_367_116_812_766_725);
    assert!(!first.cusp);

    let second = hedin_lundqvist_self_energy(5.0, 0.8)?;
    assert_real_close(second.real, -0.205_217_998_385_377_2);
    assert_real_close(second.imaginary, -0.062_054_667_501_585_545);
    assert!(second.cusp);

    let third = hedin_lundqvist_self_energy(0.15, 3.5)?;
    assert_real_close(third.real, -4.207_016_089_629_206);
    assert_real_close(third.imaginary, -0.000_000_000_589_933_275_068_458_5);
    Ok(())
}

#[test]
fn exchange_helpers_reject_invalid_inputs() {
    assert!(matches!(
        dirac_hara_exchange_potential(0.0, 1.0),
        Err(ExchangeError::NonPositiveInput { name: "rs", .. })
    ));
    assert!(matches!(
        hedin_lundqvist_ffq(0.0, 1.0, 1.0, 1.0, 1.0),
        Err(ExchangeError::NonPositiveInput { name: "q", .. })
    ));
    assert!(matches!(
        von_barth_hedin_potential(1.0, -0.1),
        Err(ExchangeError::NegativeInput { name: "xmag", .. })
    ));
    assert!(matches!(
        perdew_zunger_vxc(0.0),
        Err(ExchangeError::NonPositiveInput { name: "rs", .. })
    ));
    assert!(matches!(
        perrot_dharma_wardana_vxc(1.0, -0.1),
        Err(ExchangeError::NegativeInput { name: "temp", .. })
    ));
    assert!(matches!(
        perrot_dharma_wardana_reduced_vxc(1.0, -0.1),
        Err(ExchangeError::NegativeInput { name: "t", .. })
    ));
    assert!(matches!(
        karasiev_sjostrom_dufty_trickey_vxc(1.0, -0.1),
        Err(ExchangeError::NegativeInput { name: "temp", .. })
    ));
    assert!(matches!(
        karasiev_sjostrom_dufty_trickey_free_energy(1.0, -0.1, KsdTSpin::Unpolarized,),
        Err(ExchangeError::NegativeInput { name: "t", .. })
    ));
    assert!(matches!(
        quinn_imaginary_self_energy(0.0, 1.0, 1.0, 1.0),
        Err(ExchangeError::NonPositiveInput { name: "x", .. })
    ));
    assert!(matches!(
        hedin_lundqvist_imaginary_self_energy(1.0, 0.0),
        Err(ExchangeError::NonPositiveInput { name: "xk", .. })
    ));
    assert!(matches!(
        hedin_lundqvist_self_energy(0.0, 1.0),
        Err(ExchangeError::NonPositiveInput { name: "rs", .. })
    ));
}
