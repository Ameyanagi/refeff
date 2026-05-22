use super::*;

fn assert_complex_close(actual: Complex, expected: Complex) {
    assert!(
        (actual - expected).norm() < 1.0e-12,
        "actual={actual:?}, expected={expected:?}, diff={}",
        (actual - expected).norm()
    );
}

fn fixture() -> (Vec<Real>, Vec<Complex>) {
    (
        vec![-2.0, -0.5, 0.25, 1.5, 3.0],
        vec![
            Complex::new(0.0, -0.2),
            Complex::new(1.0, 0.3),
            Complex::new(0.25, 0.8),
            Complex::new(2.0, -0.1),
            Complex::new(1.5, 0.5),
        ],
    )
}

fn xscorratan_fixture() -> (
    Array1<Complex>,
    Array1<Complex>,
    Array1<Real>,
    Array1<Complex>,
) {
    (
        Array1::from_vec(vec![
            Complex::new(-0.20, 0.08),
            Complex::new(-0.05, 0.08),
            Complex::new(0.10, 0.08),
            Complex::new(0.30, 0.08),
            Complex::new(0.55, 0.08),
            Complex::new(0.10, 0.02),
            Complex::new(0.10, 0.05),
            Complex::new(0.12, 0.08),
        ]),
        Array1::from_vec(vec![
            Complex::new(0.40, -0.08),
            Complex::new(0.55, 0.02),
            Complex::new(0.73, 0.06),
            Complex::new(0.95, -0.01),
            Complex::new(1.10, 0.03),
            Complex::new(0.90, 0.04),
            Complex::new(0.82, 0.02),
            Complex::new(0.76, 0.01),
        ]),
        Array1::from_vec(vec![1.20, 0.85, 1.05, 0.95, 1.10, 0.70, 0.60, 0.50]),
        Array1::from_vec(vec![
            Complex::new(0.10, 0.04),
            Complex::new(-0.03, 0.05),
            Complex::new(0.08, -0.02),
            Complex::new(0.15, 0.03),
            Complex::new(-0.05, 0.06),
            Complex::new(0.02, 0.01),
            Complex::new(-0.04, 0.02),
            Complex::new(0.06, -0.03),
        ]),
    )
}

#[test]
fn conv1_matches_feff_reference() -> Result<(), ConvolutionError> {
    let (omega, spectrum) = fixture();

    let value = conv1(omega[1], omega[2], spectrum[1], spectrum[2], 0.75, 0.2)?;

    assert_complex_close(
        value,
        Complex::new(0.11548114784993968, 0.137_468_645_861_804_9),
    );
    Ok(())
}

#[test]
fn conv_matches_feff_reference() -> Result<(), ConvolutionError> {
    let (omega, spectrum) = fixture();
    let values = conv(&omega, &spectrum, 0.2)?;
    let expected = [
        Complex::new(0.11859251775642239, -0.02108887292319812),
        Complex::new(0.7780612100410474, 0.3262922950470956),
        Complex::new(0.565152671501454, 0.5900533350655084),
        Complex::new(1.6561626472085706, 0.10570908207293085),
        Complex::new(1.4178395137621587, 0.5320419613503408),
    ];

    for (&actual, expected) in values.iter().zip(expected) {
        assert_complex_close(actual, expected);
    }
    Ok(())
}

#[test]
fn conv_in_place_replaces_spectrum_values() -> Result<(), ConvolutionError> {
    let (omega, mut spectrum) = fixture();

    conv_in_place(&omega, &mut spectrum, 0.2)?;

    assert_complex_close(
        spectrum[0],
        Complex::new(0.11859251775642239, -0.02108887292319812),
    );
    assert_complex_close(
        spectrum[4],
        Complex::new(1.4178395137621587, 0.5320419613503408),
    );
    Ok(())
}

#[test]
fn ff2x_excitation_convolve_matches_feff_exconv_reference() -> Result<(), ConvolutionError> {
    let energy = Array1::from_vec(vec![-0.40, -0.12, 0.00, 0.18, 0.45, 0.90, 1.35, 1.95]);
    let xmu = Array1::from_vec(vec![0.20, 0.34, 0.50, 0.72, 0.95, 1.10, 1.04, 0.88]);

    let convolved = ff2x_excitation_convolve(Ff2xExcitationConvolutionInput {
        energy: energy.view(),
        xmu: xmu.view(),
        fermi_energy: 0.05,
        amplitude_reduction: 0.72,
        relaxation_energy: 0.18,
        plasmon_frequency: 0.55,
    })?;

    let expected = [
        0.144,
        0.244_800_000_000_000_02,
        0.36,
        0.551_374_774_249_927_1,
        0.786_619_177_385_294,
        0.988_662_602_795_670_4,
        0.996_793_437_494_949_4,
        0.892_133_865_612_567,
    ];
    for (&actual, expected) in convolved.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-14 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }
    Ok(())
}

#[test]
fn ff2x_excitation_convolve_matches_second_feff_exconv_reference() -> Result<(), ConvolutionError> {
    let energy = Array1::from_vec(vec![-0.30, -0.05, 0.08, 0.21, 0.50, 0.80, 1.20, 1.70]);
    let xmu = Array1::from_vec(vec![1.00, 0.92, 0.83, 0.70, 0.58, 0.54, 0.61, 0.75]);

    let convolved = ff2x_excitation_convolve(Ff2xExcitationConvolutionInput {
        energy: energy.view(),
        xmu: xmu.view(),
        fermi_energy: -0.02,
        amplitude_reduction: 0.35,
        relaxation_energy: 0.22,
        plasmon_frequency: 0.30,
    })?;

    let expected = [
        0.35,
        0.322,
        0.433_980_387_834_570_5,
        0.500_447_825_235_705_8,
        0.547_677_351_474_302_1,
        0.543_921_711_601_378_6,
        0.584_511_611_035_726_7,
        0.696_642_813_433_018,
    ];
    for (&actual, expected) in convolved.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-14 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }
    Ok(())
}

#[test]
fn ff2x_atan_correction_matches_feff_xscorratan_reference() -> Result<(), ConvolutionError> {
    let (energy, xsec, xsnorm, chia) = xscorratan_fixture();

    let correction = ff2x_atan_correction(Ff2xAtanCorrectionInput {
        spectroscopy: 1,
        energy: energy.view(),
        horizontal_len: 5,
        fermi_index: 2,
        xsec: xsec.view(),
        xsnorm: xsnorm.view(),
        chia: chia.view(),
        real_correction: 0.0,
        imaginary_correction: 0.0,
    })?;

    let expected = [
        Complex::new(-0.499_416_619_436_505_95, 0.030_733_330_426_861_907),
        Complex::new(-0.485_918_596_136_024_06, -0.057_902_597_251_671_115),
        Complex::new(-0.527_133_064_767_452_6, -0.025_255_761_088_366_892),
        Complex::new(-0.076_042_902_345_822_37, -0.001_287_683_014_551_682_8),
        Complex::new(-0.030_853_890_139_401_412, -0.002_834_424_357_303_861_3),
    ];
    for (&actual, expected) in correction.iter().zip(expected) {
        assert_complex_close(actual, expected);
    }
    Ok(())
}

#[test]
fn ff2x_atan_correction_matches_feff_xscorratan_emission_reference() -> Result<(), ConvolutionError>
{
    let (energy, xsec, xsnorm, chia) = xscorratan_fixture();

    let correction = ff2x_atan_correction(Ff2xAtanCorrectionInput {
        spectroscopy: 2,
        energy: energy.view(),
        horizontal_len: 5,
        fermi_index: 2,
        xsec: xsec.view(),
        xsnorm: xsnorm.view(),
        chia: chia.view(),
        real_correction: 0.0,
        imaginary_correction: 0.0,
    })?;

    let expected = [
        Complex::new(-0.020_583_380_563_494_07, 0.001_266_669_573_138_094),
        Complex::new(-0.038_581_403_863_976_016, -0.004_597_402_748_328_885),
        Complex::new(-0.286_866_935_232_547_36, -0.013_744_238_911_633_101),
        Complex::new(-1.016_457_097_654_177_6, -0.017_212_316_985_448_31),
        Complex::new(-1.014_146_109_860_598_8, -0.093_165_575_642_696_15),
    ];
    for (&actual, expected) in correction.iter().zip(expected) {
        assert_complex_close(actual, expected);
    }
    Ok(())
}

#[test]
fn ff2x_atan_correction_matches_feff_xscorratan_real_shift_reference()
-> Result<(), ConvolutionError> {
    let (energy, xsec, xsnorm, chia) = xscorratan_fixture();

    let correction = ff2x_atan_correction(Ff2xAtanCorrectionInput {
        spectroscopy: 1,
        energy: energy.view(),
        horizontal_len: 5,
        fermi_index: 2,
        xsec: xsec.view(),
        xsnorm: xsnorm.view(),
        chia: chia.view(),
        real_correction: 0.03,
        imaginary_correction: 0.0,
    })?;

    let expected = [
        Complex::new(-0.497_312_650_460_951_86, 0.030_603_855_412_981_65),
        Complex::new(-0.478_036_888_055_366_54, -0.056_963_404_201_068_456),
        Complex::new(-0.343_524_987_872_821_3, -0.016_458_813_915_282_592),
        Complex::new(-0.065_454_696_779_511_84, -0.001_108_386_169_721_710_5),
        Complex::new(-0.028_852_105_893_751_454, -0.002_650_528_388_325_492),
    ];
    for (&actual, expected) in correction.iter().zip(expected) {
        assert_complex_close(actual, expected);
    }
    Ok(())
}

#[test]
fn ff2x_excitation_convolve_preserves_xmu_when_s02_is_nearly_one() -> Result<(), ConvolutionError> {
    let energy = Array1::from_vec(vec![-0.30, -0.05, 0.08, 0.21]);
    let xmu = Array1::from_vec(vec![1.00, 0.92, 0.83, 0.70]);

    let convolved = ff2x_excitation_convolve(Ff2xExcitationConvolutionInput {
        energy: energy.view(),
        xmu: xmu.view(),
        fermi_energy: -0.02,
        amplitude_reduction: 1.0,
        relaxation_energy: 0.22,
        plasmon_frequency: 0.30,
    })?;

    assert_eq!(convolved, xmu);
    Ok(())
}

#[test]
fn rejects_invalid_inputs() {
    assert!(matches!(
        conv(&[1.0], &[Complex::new(1.0, 0.0)], 0.2),
        Err(ConvolutionError::InsufficientPoints { .. })
    ));
    assert!(matches!(
        conv(
            &[1.0, 1.0],
            &[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)],
            0.2
        ),
        Err(ConvolutionError::DuplicateEndpointEnergy)
    ));
    assert!(matches!(
        conv1(
            0.0,
            1.0,
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            0.5,
            0.0
        ),
        Err(ConvolutionError::InvalidWidth { .. })
    ));
    let energy = Array1::from_vec(vec![0.0, 0.2, 0.1]);
    let xmu = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    assert!(matches!(
        ff2x_excitation_convolve(Ff2xExcitationConvolutionInput {
            energy: energy.view(),
            xmu: xmu.view(),
            fermi_energy: 0.05,
            amplitude_reduction: 0.5,
            relaxation_energy: 0.2,
            plasmon_frequency: 0.3,
        }),
        Err(ConvolutionError::ExcitationNonIncreasingEnergy { row: 2, .. })
    ));
    let energy = Array1::from_vec(vec![0.0, 0.2, 0.4]);
    assert!(matches!(
        ff2x_excitation_convolve(Ff2xExcitationConvolutionInput {
            energy: energy.view(),
            xmu: xmu.view(),
            fermi_energy: -0.1,
            amplitude_reduction: 0.5,
            relaxation_energy: 0.2,
            plasmon_frequency: 0.3,
        }),
        Err(ConvolutionError::ExcitationFermiOutOfRange { .. })
    ));
    let (energy, xsec, xsnorm, chia) = xscorratan_fixture();
    assert!(matches!(
        ff2x_atan_correction(Ff2xAtanCorrectionInput {
            spectroscopy: 1,
            energy: energy.view(),
            horizontal_len: 0,
            fermi_index: 0,
            xsec: xsec.view(),
            xsnorm: xsnorm.view(),
            chia: chia.view(),
            real_correction: 0.0,
            imaginary_correction: 0.0,
        }),
        Err(ConvolutionError::AtanInvalidHorizontalLength { .. })
    ));
    assert!(matches!(
        ff2x_atan_correction(Ff2xAtanCorrectionInput {
            spectroscopy: 1,
            energy: energy.view(),
            horizontal_len: 5,
            fermi_index: 5,
            xsec: xsec.view(),
            xsnorm: xsnorm.view(),
            chia: chia.view(),
            real_correction: 0.0,
            imaginary_correction: 0.0,
        }),
        Err(ConvolutionError::AtanFermiIndexOutOfRange { .. })
    ));
}
