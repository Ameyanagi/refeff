use ndarray::{Array1, array};
use num_complex::Complex64;
use refeff_core::DmdwPoleWeightedA2f;

use crate::Result;

use super::*;

const DMDW_EGRID_INFO: &str = "\
#  Energies printed in meV
#  lowE   -125.000 highE    175.000
#  dE  =      0.030 w0 =     15.000
#  Ek  =      5.000 --> E =      4.990
";

const DMDW_SPECTRAL_INFO: &str = "\
Gamma_k =   5.0000000000D-03
epk = E_k - ReSE(E_k) =   3.3333333333D-01

atot    =  ( -1.2500000000D-01,  2.5000000000D-02)
Zk      =  (  8.8000000000D-01, -2.2000000000D-02)
";

const DMDW_A2F_INFO: &str = "\
# DMDW Option           2
# Displacement Option           1
# Lanczos Order           3
#
# Lanczos Pole in Thz/weight PHDOS
  1.5915494309D+00  2.0000000000D-01
  3.1830988618D+00  3.0000000000D-01
  4.7746482928D+00  5.0000000000D-01
# norm  1.2500000000D+01

Pole/weight a2f in eV/Arb
  6.5821192800D-03  1.0000000000D-01
  1.3164238560D-02  2.0000000000D-01
  1.9746357840D-02  3.0000000000D-01
lambda =  7.6000000000D+01
w0 =  1.5358278320D-02
";

const DMDW_RESE_DAT: &str = "\
#  Real part of the Self-energy
 -1.5000000000D-01  2.5000000000D-03
  0.0000000000D+00  0.0000000000D+00
  1.5000000000D-01 -2.5000000000D-03
";

const DMDW_AKW_DAT: &str = "\
# norm =   1.2345000000D+00
# w [meV], mag, ph, re, im
      -150.0000000000        0.0100000000       -1.5700000000        0.0000000000       -0.0100000000
         0.0000000000        0.5000000000        0.0000000000        0.5000000000        0.0000000000
       150.0000000000        0.0100000000        1.5700000000        0.0000000000        0.0100000000
";

#[test]
fn parses_and_renders_dmdw_a2f_info() -> Result<()> {
    let parsed = parse_dmdw_a2f_info(DMDW_A2F_INFO)?;
    assert_eq!(parsed.calculation_type, 2);
    assert_eq!(parsed.displacement_option, 1);
    assert_eq!(parsed.lanczos_order, 3);
    assert_eq!(
        parsed.lanczos_frequency_thz,
        array![1.591_549_430_9, 3.183_098_861_8, 4.774_648_292_8]
    );
    assert_eq!(parsed.lanczos_weight, array![0.2, 0.3, 0.5]);
    assert_eq!(parsed.normalization, 12.5);
    assert_eq!(
        parsed.pole_energy_ev,
        array![0.006_582_119_28, 0.013_164_238_56, 0.019_746_357_84]
    );
    assert_eq!(parsed.pole_weight, array![0.1, 0.2, 0.3]);
    assert_eq!(parsed.mass_enhancement, 76.0);
    assert_eq!(parsed.characteristic_energy_ev, 0.015_358_278_32);

    let rendered = dmdw_a2f_info_string(&parsed)?;
    let reparsed = parse_dmdw_a2f_info(&rendered)?;
    assert_a2f_info_close(&reparsed, &parsed);
    Ok(())
}

#[test]
fn builds_dmdw_a2f_info_from_core_diagnostic() -> Result<()> {
    let diagnostic = DmdwPoleWeightedA2f {
        lanczos_frequency_thz: array![1.0, 2.0],
        lanczos_weight: array![0.25, 0.75],
        normalization: 3.0,
        pole_energy_ev: array![0.01, 0.02],
        pole_weight: array![0.1, 0.4],
        mass_enhancement: 60.0,
        characteristic_energy_ev: 0.018,
    };

    let data = dmdw_a2f_info_from_pole_weighted(2, 1, 2, &diagnostic)?;
    assert_eq!(data.calculation_type, 2);
    assert_eq!(data.displacement_option, 1);
    assert_eq!(data.lanczos_order, 2);
    assert_eq!(data.lanczos_frequency_thz, diagnostic.lanczos_frequency_thz);
    assert_eq!(data.lanczos_weight, diagnostic.lanczos_weight);
    assert_eq!(data.normalization, diagnostic.normalization);
    assert_eq!(data.pole_energy_ev, diagnostic.pole_energy_ev);
    assert_eq!(data.pole_weight, diagnostic.pole_weight);
    assert_eq!(data.mass_enhancement, diagnostic.mass_enhancement);
    assert_eq!(
        data.characteristic_energy_ev,
        diagnostic.characteristic_energy_ev
    );
    Ok(())
}

#[test]
fn parses_and_renders_dmdw_egrid_info() -> Result<()> {
    let parsed = parse_dmdw_egrid_info(DMDW_EGRID_INFO)?;
    assert_eq!(
        parsed,
        DmdwEnergyGridInfo {
            low_energy_mev: -125.0,
            high_energy_mev: 175.0,
            step_mev: 0.03,
            characteristic_energy_mev: 15.0,
            electron_energy_mev: 5.0,
            selected_energy_mev: 4.99,
        }
    );

    let rendered = dmdw_egrid_info_string(&parsed)?;
    let reparsed = parse_dmdw_egrid_info(&rendered)?;
    assert_eq!(reparsed, parsed);
    Ok(())
}

#[test]
fn parses_and_renders_dmdw_spectral_info() -> Result<()> {
    let parsed = parse_dmdw_spectral_info(DMDW_SPECTRAL_INFO)?;
    assert_close(parsed.gamma, 0.005);
    assert_close(parsed.effective_electron_energy, 0.333_333_333_3);
    assert_complex_close(
        parsed.total_cumulant_derivative,
        Complex64::new(-0.125, 0.025),
    );
    assert_complex_close(parsed.quasiparticle_weight, Complex64::new(0.88, -0.022));

    let rendered = dmdw_spectral_info_string(&parsed)?;
    let reparsed = parse_dmdw_spectral_info(&rendered)?;
    assert_spectral_info_close(&reparsed, &parsed);
    Ok(())
}

#[test]
fn parses_and_renders_dmdw_self_energy_dat() -> Result<()> {
    let parsed = parse_dmdw_self_energy_dat(DMDW_RESE_DAT)?;
    assert_eq!(parsed.header_lines, vec!["#  Real part of the Self-energy"]);
    assert_eq!(parsed.energy_ev, array![-0.15, 0.0, 0.15]);
    assert_eq!(parsed.value_ev, array![0.0025, 0.0, -0.0025]);

    let rendered = dmdw_self_energy_dat_string(&parsed)?;
    let reparsed = parse_dmdw_self_energy_dat(&rendered)?;
    assert_eq!(reparsed, parsed);
    Ok(())
}

#[test]
fn parses_and_renders_dmdw_akw_dat() -> Result<()> {
    let parsed = parse_dmdw_akw_dat(DMDW_AKW_DAT)?;
    assert_eq!(parsed.normalization, Some(1.2345));
    assert_eq!(parsed.energy_mev, array![-150.0, 0.0, 150.0]);
    assert_eq!(parsed.magnitude, array![0.01, 0.5, 0.01]);
    assert_eq!(parsed.phase, array![-1.57, 0.0, 1.57]);
    assert_eq!(parsed.real, array![0.0, 0.5, 0.0]);
    assert_eq!(parsed.imaginary, array![-0.01, 0.0, 0.01]);

    let rendered = dmdw_akw_dat_string(&parsed)?;
    let reparsed = parse_dmdw_akw_dat(&rendered)?;
    assert_eq!(reparsed, parsed);
    Ok(())
}

#[test]
fn rejects_invalid_dmdw_self_energy_sidecars() {
    assert!(parse_dmdw_a2f_info("1.0 2.0\n").is_err());
    assert!(
        dmdw_a2f_info_string(&DmdwA2fInfoData {
            calculation_type: 2,
            displacement_option: 1,
            lanczos_order: 3,
            lanczos_frequency_thz: array![1.0],
            lanczos_weight: array![1.0],
            normalization: 0.0,
            pole_energy_ev: array![0.1],
            pole_weight: array![0.2],
            mass_enhancement: 4.0,
            characteristic_energy_ev: 0.1,
        })
        .is_err()
    );
    assert!(parse_dmdw_egrid_info("# too short\n").is_err());
    assert!(
        dmdw_egrid_info_string(&DmdwEnergyGridInfo {
            low_energy_mev: 1.0,
            high_energy_mev: 0.0,
            step_mev: 0.1,
            characteristic_energy_mev: 1.0,
            electron_energy_mev: 0.0,
            selected_energy_mev: 0.0,
        })
        .is_err()
    );
    assert!(parse_dmdw_spectral_info("Gamma_k = 0.005\n").is_err());
    assert!(
        dmdw_spectral_info_string(&DmdwSpectralInfoData {
            gamma: 0.0,
            effective_electron_energy: 0.0,
            total_cumulant_derivative: Complex64::new(0.0, 0.0),
            quasiparticle_weight: Complex64::new(1.0, 0.0),
        })
        .is_err()
    );
    assert!(parse_dmdw_self_energy_dat("1.0 2.0 3.0\n").is_err());
    assert!(parse_dmdw_self_energy_dat("1.0 NaN\n").is_err());
    assert!(parse_dmdw_akw_dat("1.0 2.0\n").is_err());
    assert!(parse_dmdw_akw_dat("1.0 2.0 3.0 4.0 inf\n").is_err());

    let bad = DmdwAkwDatData {
        normalization: None,
        energy_mev: array![0.0],
        magnitude: array![1.0, 2.0],
        phase: array![0.0],
        real: array![1.0],
        imaginary: array![0.0],
    };
    assert!(dmdw_akw_dat_string(&bad).is_err());
}

fn assert_a2f_info_close(actual: &DmdwA2fInfoData, expected: &DmdwA2fInfoData) {
    assert_eq!(actual.calculation_type, expected.calculation_type);
    assert_eq!(actual.displacement_option, expected.displacement_option);
    assert_eq!(actual.lanczos_order, expected.lanczos_order);
    assert_array_close(
        &actual.lanczos_frequency_thz,
        &expected.lanczos_frequency_thz,
    );
    assert_array_close(&actual.lanczos_weight, &expected.lanczos_weight);
    assert_close(actual.normalization, expected.normalization);
    assert_array_close(&actual.pole_energy_ev, &expected.pole_energy_ev);
    assert_array_close(&actual.pole_weight, &expected.pole_weight);
    assert_close(actual.mass_enhancement, expected.mass_enhancement);
    assert_close(
        actual.characteristic_energy_ev,
        expected.characteristic_energy_ev,
    );
}

fn assert_array_close(actual: &Array1<f64>, expected: &Array1<f64>) {
    assert_eq!(actual.len(), expected.len());
    for (&left, &right) in actual.iter().zip(expected.iter()) {
        assert_close(left, right);
    }
}

fn assert_spectral_info_close(actual: &DmdwSpectralInfoData, expected: &DmdwSpectralInfoData) {
    assert_close(actual.gamma, expected.gamma);
    assert_close(
        actual.effective_electron_energy,
        expected.effective_electron_energy,
    );
    assert_complex_close(
        actual.total_cumulant_derivative,
        expected.total_cumulant_derivative,
    );
    assert_complex_close(actual.quasiparticle_weight, expected.quasiparticle_weight);
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-9,
        "actual {actual} differed from expected {expected}"
    );
}

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert_close(actual.re, expected.re);
    assert_close(actual.im, expected.im);
}
