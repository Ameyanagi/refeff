//! Debye and Einstein-model cumulant helpers ported from FEFF.
//!
//! This module starts with `DEBYE/sigm3.f90`, the correlated Einstein model
//! with a Morse potential used for first and third cumulant estimates.

use crate::Real;

const BOHR_ANGSTROM: Real = 0.529_177_249;
const HBAR: Real = 1.054_572_7e-34_f32 as Real;
const ATOMIC_MASS_UNIT: Real = 1.660_54e-27_f32 as Real;
const BOLTZMANN: Real = 1.380_658e-23_f32 as Real;
const DEBYE_CORRELATION_FACTOR: Real = 48.508_46_f32 as Real;
const DEBYE_ROMBERG_TOLERANCE: Real = 1.0e-5;
const DEBYE_ROMBERG_MAX_ITERATIONS: usize = 10;

const FEFF_ATOMIC_WEIGHTS: [f32; 139] = [
    1.0079, 4.0026, 6.941, 9.0122, 10.81, 12.01, 14.007, 15.999, 18.998, 20.18, 22.9898, 24.305,
    26.982, 28.086, 30.974, 32.064, 35.453, 39.948, 39.09, 40.08, 44.956, 47.90, 50.942, 52.00,
    54.938, 55.85, 58.93, 58.71, 63.55, 65.38, 69.72, 72.59, 74.922, 78.96, 79.91, 83.80, 85.47,
    87.62, 88.91, 91.22, 92.91, 95.94, 98.91, 101.07, 102.90, 106.40, 107.87, 112.40, 114.82,
    118.69, 121.75, 127.60, 126.90, 131.30, 132.91, 137.34, 138.91, 140.12, 140.91, 144.24, 145.0,
    150.35, 151.96, 157.25, 158.92, 162.50, 164.93, 167.26, 168.93, 173.04, 174.97, 178.49, 180.95,
    183.85, 186.2, 190.20, 192.22, 195.09, 196.97, 200.59, 204.37, 207.19, 208.98, 210.0, 210.0,
    222.0, 223.0, 226.0, 227.0, 232.04, 231.0, 238.03, 237.05, 244.0, 243.0, 247.0, 247.0, 251.0,
    252.0, 257.0, 258.0, 259.0, 266.0, 267.0, 268.0, 269.0, 270.0, 269.0, 278.0, 281.0, 282.0,
    285.0, 286.0, 289.0, 289.0, 293.0, 294.0, 294.0, 315.0, 320.0, 330.0, 334.0, 337.0, 340.0,
    344.0, 347.0, 350.0, 354.0, 357.0, 361.0, 364.0, 367.0, 371.0, 374.0, 378.0, 381.0, 385.0,
    388.0, 392.0,
];

/// First and third cumulants from FEFF `sigm3`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MorseCumulants {
    /// FEFF `sig1`: first cumulant.
    pub first: Real,
    /// FEFF `sig3`: third cumulant.
    pub third: Real,
    /// FEFF mutates `alphat` from inverse angstrom to inverse bohr; Rust returns
    /// the scaled value explicitly.
    pub scaled_thermal_expansion: Real,
}

/// First and third cumulants from FEFF `sigte3`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalExpansionCumulants {
    /// FEFF `sig1`: first cumulant.
    pub first: Real,
    /// FEFF `sig3`: third cumulant.
    pub third: Real,
}

/// Correlated Debye-model displacement correlation from FEFF `corrfn`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DebyeCorrelation {
    /// Correlation value returned as FEFF `cij`.
    pub value: Real,
    /// Relative Romberg estimate used by FEFF `bingrt`.
    pub estimated_error: Real,
    /// Number of binary-refinement iterations used by the integration.
    pub iterations: usize,
}

/// Error returned by Debye/Einstein cumulant helpers.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum DebyeError {
    /// Inputs must be finite real values.
    #[error("Debye input {name} must be finite, got {value}")]
    NonFinite { name: &'static str, value: Real },
    /// Inputs used as scales must be strictly positive.
    #[error("Debye input {name} must be positive, got {value}")]
    NonPositive { name: &'static str, value: Real },
    /// Inputs used as nonnegative values must be zero or positive.
    #[error("Debye input {name} must be nonnegative, got {value}")]
    Negative { name: &'static str, value: Real },
    /// FEFF's periodic table covers atomic numbers 1 through 139.
    #[error("Debye atomic number must be in 1..=139, got {z}")]
    InvalidAtomicNumber { z: usize },
    /// A computed output became non-finite.
    #[error("Debye output {name} must be finite, got {value}")]
    NonFiniteOutput { name: &'static str, value: Real },
    /// FEFF Romberg integration did not converge within the configured limit.
    #[error(
        "{routine} did not converge after {iterations} iterations; estimated error {estimated_error}"
    )]
    IntegrationDidNotConverge {
        routine: &'static str,
        iterations: usize,
        estimated_error: Real,
    },
}

/// Port of FEFF `sigm3`: correlated Einstein-model Morse cumulants.
///
/// `mean_square_relative_displacement` is FEFF `sig2`, `temperature` is `tk`,
/// `thermal_expansion` is `alphat` in inverse angstrom, and
/// `einstein_temperature` is `thetae`. FEFF stores several intermediates as
/// single precision `real`; this port keeps those roundings to match the
/// reference values.
pub fn morse_einstein_cumulants(
    mean_square_relative_displacement: Real,
    temperature: Real,
    thermal_expansion: Real,
    einstein_temperature: Real,
) -> Result<MorseCumulants, DebyeError> {
    ensure_positive("sig2", mean_square_relative_displacement)?;
    ensure_positive("tk", temperature)?;
    ensure_finite("alphat", thermal_expansion)?;
    ensure_positive("thetae", einstein_temperature)?;

    let scaled_thermal_expansion = thermal_expansion * BOHR_ANGSTROM;
    let z = to_feff_real((-einstein_temperature / temperature).exp());
    let occupation_ratio = to_feff_real(((1.0_f32 - z as f32) / (1.0_f32 + z as f32)) as Real);
    let sig02 = to_feff_real(occupation_ratio * mean_square_relative_displacement);
    let sig01 = to_feff_real(scaled_thermal_expansion * sig02 * 0.75);
    let first = sig01 * mean_square_relative_displacement / sig02;
    let third = (2.0
        - ((4.0_f32 / 3.0_f32) as Real) * (sig02 / mean_square_relative_displacement).powi(2))
        * first
        * mean_square_relative_displacement;

    ensure_finite_output("sig1", first)?;
    ensure_finite_output("sig3", third)?;
    ensure_finite_output("alphat", scaled_thermal_expansion)?;

    Ok(MorseCumulants {
        first,
        third,
        scaled_thermal_expansion,
    })
}

/// Port of FEFF `sigte3`: thermal-expansion first and third cumulants.
///
/// `central_atomic_number` and `neighbor_atomic_number` are FEFF `iz1` and
/// `iz2`; `mean_square_relative_displacement` is `sig2`; `thermal_expansion`
/// is `alphat`; `debye_temperature` is `thetad`; and
/// `effective_distance_angstrom` is FEFF's single-precision `reff`.
pub fn thermal_expansion_cumulants(
    central_atomic_number: usize,
    neighbor_atomic_number: usize,
    mean_square_relative_displacement: Real,
    thermal_expansion: Real,
    debye_temperature: Real,
    effective_distance_angstrom: Real,
) -> Result<ThermalExpansionCumulants, DebyeError> {
    let central_mass = atomic_weight(central_atomic_number)? * ATOMIC_MASS_UNIT;
    let neighbor_mass = atomic_weight(neighbor_atomic_number)? * ATOMIC_MASS_UNIT;
    ensure_positive("sig2", mean_square_relative_displacement)?;
    ensure_finite("alphat", thermal_expansion)?;
    ensure_positive("thetad", debye_temperature)?;
    ensure_positive("reff", effective_distance_angstrom)?;

    let reff = to_feff_real(effective_distance_angstrom);
    let reduced_mass = 1.0 / (1.0 / central_mass + 1.0 / neighbor_mass);
    let omega = (2.0 * BOLTZMANN * debye_temperature) / (3.0 * HBAR);
    let spring_constant = reduced_mass * omega.powi(2);
    let cubic_force_constant =
        spring_constant.powi(2) * reff * thermal_expansion / (3.0 * BOLTZMANN);
    let sig02 = HBAR * omega / spring_constant;
    let first = -3.0 * (cubic_force_constant / spring_constant) * mean_square_relative_displacement;
    let third = (2.0
        - ((4.0_f32 / 3.0_f32) as Real) * (sig02 / mean_square_relative_displacement).powi(2))
        * first
        * mean_square_relative_displacement;

    ensure_finite_output("sig1", first)?;
    ensure_finite_output("sig3", third)?;

    Ok(ThermalExpansionCumulants { first, third })
}

/// Port of FEFF `corrfn`: quantum Debye displacement correlation.
///
/// `distance_angstrom` is FEFF `rij`, `debye_temperature` is `thetad`,
/// `temperature` is `tk`, the atomic numbers are `iz1`/`iz2`, and
/// `average_wigner_seitz_radius_bohr` is `rsavg`.
pub fn quantum_debye_correlation(
    distance_angstrom: Real,
    debye_temperature: Real,
    temperature: Real,
    first_atomic_number: usize,
    second_atomic_number: usize,
    average_wigner_seitz_radius_bohr: Real,
) -> Result<DebyeCorrelation, DebyeError> {
    debye_correlation(DebyeCorrelationInput {
        routine: "corrfn",
        distance_angstrom,
        debye_temperature,
        temperature,
        first_atomic_number,
        second_atomic_number,
        average_wigner_seitz_radius_bohr,
        integrand: quantum_debye_integrand,
    })
}

/// Port of FEFF `corrfn2`: classical Debye displacement correlation.
pub fn classical_debye_correlation(
    distance_angstrom: Real,
    debye_temperature: Real,
    temperature: Real,
    first_atomic_number: usize,
    second_atomic_number: usize,
    average_wigner_seitz_radius_bohr: Real,
) -> Result<DebyeCorrelation, DebyeError> {
    debye_correlation(DebyeCorrelationInput {
        routine: "corrfn2",
        distance_angstrom,
        debye_temperature,
        temperature,
        first_atomic_number,
        second_atomic_number,
        average_wigner_seitz_radius_bohr,
        integrand: classical_debye_integrand,
    })
}

struct DebyeCorrelationInput {
    routine: &'static str,
    distance_angstrom: Real,
    debye_temperature: Real,
    temperature: Real,
    first_atomic_number: usize,
    second_atomic_number: usize,
    average_wigner_seitz_radius_bohr: Real,
    integrand: fn(Real, Real, Real) -> Real,
}

fn debye_correlation(input: DebyeCorrelationInput) -> Result<DebyeCorrelation, DebyeError> {
    let DebyeCorrelationInput {
        routine,
        distance_angstrom,
        debye_temperature,
        temperature,
        first_atomic_number,
        second_atomic_number,
        average_wigner_seitz_radius_bohr,
        integrand,
    } = input;

    ensure_nonnegative("rij", distance_angstrom)?;
    ensure_positive("thetad", debye_temperature)?;
    ensure_nonnegative("tk", temperature)?;
    ensure_positive("rsavg", average_wigner_seitz_radius_bohr)?;
    let first_mass = atomic_weight(first_atomic_number)?;
    let second_mass = atomic_weight(second_atomic_number)?;

    let y_inverse = temperature / debye_temperature;
    let debye_wave_number = (9.0 * std::f64::consts::PI / 2.0).powf(1.0 / 3.0)
        / (average_wigner_seitz_radius_bohr * BOHR_ANGSTROM);
    let x = debye_wave_number * distance_angstrom;
    let factor = ((3.0_f32 / 2.0_f32) as Real) * DEBYE_CORRELATION_FACTOR
        / (debye_temperature * (first_mass * second_mass).sqrt());
    let (integral, estimated_error, iterations) =
        integrate_debye_romberg(routine, |w| integrand(w, x, y_inverse))?;
    let value = factor * integral;
    ensure_finite_output(routine, value)?;
    Ok(DebyeCorrelation {
        value,
        estimated_error,
        iterations,
    })
}

fn quantum_debye_integrand(w: Real, x: Real, y_inverse: Real) -> Real {
    if w < 1.0e-20 {
        return 2.0 * y_inverse;
    }
    let factor = if x > 0.0 { (w * x).sin() / x } else { w };
    let exp_term = (-w / y_inverse).exp();
    factor * (1.0 + exp_term) / (1.0 - exp_term)
}

fn classical_debye_integrand(w: Real, x: Real, y_inverse: Real) -> Real {
    if w < 1.0e-20 {
        return 2.0 * y_inverse;
    }
    let factor = if x > 0.0 { (w * x).sin() / x } else { w };
    factor * 2.0 * y_inverse / w
}

fn integrate_debye_romberg(
    routine: &'static str,
    integrand: impl Fn(Real) -> Real,
) -> Result<(Real, Real, usize), DebyeError> {
    let mut intervals = 1;
    let mut delta = 1.0;
    let mut previous_trapezoid = (integrand(0.0) + integrand(1.0)) / 2.0;
    let mut previous_extrapolated = previous_trapezoid;
    let mut estimated_error = Real::INFINITY;

    for iteration in 1..=DEBYE_ROMBERG_MAX_ITERATIONS {
        delta /= 2.0;
        let midpoint_sum = (1..=intervals)
            .map(|index| {
                let z = (2 * index - 1) as Real * delta;
                integrand(z)
            })
            .sum::<Real>();
        let trapezoid = previous_trapezoid / 2.0 + delta * midpoint_sum;
        let extrapolated = (4.0 * trapezoid - previous_trapezoid) / 3.0;
        estimated_error = relative_error(extrapolated, previous_extrapolated);
        if estimated_error < DEBYE_ROMBERG_TOLERANCE {
            return Ok((extrapolated, estimated_error, iteration));
        }
        previous_trapezoid = trapezoid;
        previous_extrapolated = extrapolated;
        intervals *= 2;
    }

    Err(DebyeError::IntegrationDidNotConverge {
        routine,
        iterations: DEBYE_ROMBERG_MAX_ITERATIONS,
        estimated_error,
    })
}

fn relative_error(current: Real, previous: Real) -> Real {
    if current == 0.0 {
        if previous == 0.0 { 0.0 } else { Real::INFINITY }
    } else {
        ((current - previous) / current).abs()
    }
}

fn atomic_weight(atomic_number: usize) -> Result<Real, DebyeError> {
    if atomic_number == 0 {
        return Err(DebyeError::InvalidAtomicNumber { z: atomic_number });
    }
    FEFF_ATOMIC_WEIGHTS
        .get(atomic_number - 1)
        .map(|&weight| Real::from(weight))
        .ok_or(DebyeError::InvalidAtomicNumber { z: atomic_number })
}

fn ensure_nonnegative(name: &'static str, value: Real) -> Result<(), DebyeError> {
    ensure_finite(name, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(DebyeError::Negative { name, value })
    }
}

fn to_feff_real(value: Real) -> Real {
    (value as f32) as Real
}

fn ensure_finite(name: &'static str, value: Real) -> Result<(), DebyeError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DebyeError::NonFinite { name, value })
    }
}

fn ensure_positive(name: &'static str, value: Real) -> Result<(), DebyeError> {
    ensure_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(DebyeError::NonPositive { name, value })
    }
}

fn ensure_finite_output(name: &'static str, value: Real) -> Result<(), DebyeError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DebyeError::NonFiniteOutput { name, value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morse_einstein_cumulants_match_feff_reference() -> Result<(), DebyeError> {
        let first = morse_einstein_cumulants(0.003, 300.0, 1.0e-5, 400.0)?;
        assert_close(first.first, 1.190_648_842_682_321_3e-8);
        assert_close(first.third, 5.526_344_214_607_83e-11);
        assert_close(first.scaled_thermal_expansion, 5.291_772_49e-6);

        let second = morse_einstein_cumulants(0.0075, 800.0, 2.5e-5, 250.0)?;
        assert_close(second.first, 7.441_554_786_684_262e-8);
        assert_close(second.third, 1.098_357_016_560_439_2e-9);
        assert_close(second.scaled_thermal_expansion, 1.322_943_122_5e-5);

        let negative_alpha = morse_einstein_cumulants(0.0012, 120.0, -7.0e-6, 350.0)?;
        assert_close(negative_alpha.first, -3.333_816_545_419_16e-9);
        assert_close(negative_alpha.third, -3.706_146_208_663_239e-12);
        assert_close(negative_alpha.scaled_thermal_expansion, -3.704_240_743e-6);
        Ok(())
    }

    #[test]
    fn morse_einstein_cumulants_reject_invalid_inputs() {
        assert!(matches!(
            morse_einstein_cumulants(0.0, 300.0, 1.0e-5, 400.0),
            Err(DebyeError::NonPositive { name: "sig2", .. })
        ));
        assert!(matches!(
            morse_einstein_cumulants(0.003, Real::NAN, 1.0e-5, 400.0),
            Err(DebyeError::NonFinite { name: "tk", .. })
        ));
        assert!(matches!(
            morse_einstein_cumulants(0.003, 300.0, Real::INFINITY, 400.0),
            Err(DebyeError::NonFinite { name: "alphat", .. })
        ));
        assert!(matches!(
            morse_einstein_cumulants(0.003, 300.0, 1.0e-5, -1.0),
            Err(DebyeError::NonPositive { name: "thetae", .. })
        ));
    }

    #[test]
    fn thermal_expansion_cumulants_match_feff_reference() -> Result<(), DebyeError> {
        let copper = thermal_expansion_cumulants(29, 29, 0.003, 1.0e-5, 400.0, 2.55)?;
        assert_relative_close(copper.first, -3.563_418_839_026_406e17);
        assert_relative_close(copper.third, -2.138_051_303_415_843_5e15);

        let copper_oxygen = thermal_expansion_cumulants(29, 8, 0.0042, 1.8e-5, 650.0, 1.91)?;
        assert_relative_close(copper_oxygen.first, -7.144_230_125_822_932e17);
        assert_relative_close(copper_oxygen.third, -6.001_153_305_691_263e15);

        let carbon_hydrogen = thermal_expansion_cumulants(6, 1, 0.0015, -6.0e-6, 300.0, 1.09)?;
        assert_relative_close(carbon_hydrogen.first, 7.521_958_969_413_031e14);
        assert_relative_close(carbon_hydrogen.third, 2.256_587_690_823_909e12);
        Ok(())
    }

    #[test]
    fn thermal_expansion_cumulants_reject_invalid_inputs() {
        assert!(matches!(
            thermal_expansion_cumulants(0, 29, 0.003, 1.0e-5, 400.0, 2.55),
            Err(DebyeError::InvalidAtomicNumber { z: 0 })
        ));
        assert!(matches!(
            thermal_expansion_cumulants(29, 140, 0.003, 1.0e-5, 400.0, 2.55),
            Err(DebyeError::InvalidAtomicNumber { z: 140 })
        ));
        assert!(matches!(
            thermal_expansion_cumulants(29, 29, -0.003, 1.0e-5, 400.0, 2.55),
            Err(DebyeError::NonPositive { name: "sig2", .. })
        ));
        assert!(matches!(
            thermal_expansion_cumulants(29, 29, 0.003, 1.0e-5, 0.0, 2.55),
            Err(DebyeError::NonPositive { name: "thetad", .. })
        ));
        assert!(matches!(
            thermal_expansion_cumulants(29, 29, 0.003, 1.0e-5, 400.0, Real::NAN),
            Err(DebyeError::NonFinite { name: "reff", .. })
        ));
    }

    #[test]
    fn debye_correlations_match_feff_reference() -> Result<(), DebyeError> {
        let zero = quantum_debye_correlation(0.0, 400.0, 300.0, 29, 29, 2.7)?;
        assert_close(zero.value, 4.501_999_849_393_054e-3);
        let copper = quantum_debye_correlation(2.55, 400.0, 300.0, 29, 29, 2.7)?;
        assert_close(copper.value, 1.691_640_883_386_128e-3);
        let copper_oxygen = quantum_debye_correlation(1.91, 650.0, 120.0, 29, 8, 2.3)?;
        assert_close(copper_oxygen.value, 7.447_746_368_694_431e-4);

        let classical_zero = classical_debye_correlation(0.0, 400.0, 300.0, 29, 29, 2.7)?;
        assert_close(classical_zero.value, 4.293_628_582_101_32e-3);
        let classical_copper = classical_debye_correlation(2.55, 400.0, 300.0, 29, 29, 2.7)?;
        assert_close(classical_copper.value, 1.685_437_153_407_153e-3);
        let classical_copper_oxygen = classical_debye_correlation(1.91, 650.0, 120.0, 29, 8, 2.3)?;
        assert_close(classical_copper_oxygen.value, 6.129_399_740_209_465e-4);
        Ok(())
    }

    #[test]
    fn debye_correlations_reject_invalid_inputs() {
        assert!(matches!(
            quantum_debye_correlation(-1.0, 400.0, 300.0, 29, 29, 2.7),
            Err(DebyeError::Negative { name: "rij", .. })
        ));
        assert!(matches!(
            quantum_debye_correlation(1.0, 400.0, -1.0, 29, 29, 2.7),
            Err(DebyeError::Negative { name: "tk", .. })
        ));
        assert!(matches!(
            classical_debye_correlation(1.0, 400.0, 300.0, 29, 0, 2.7),
            Err(DebyeError::InvalidAtomicNumber { z: 0 })
        ));
    }

    fn assert_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() < 1.0e-18,
            "actual={actual} expected={expected} diff={}",
            (actual - expected).abs()
        );
    }

    fn assert_relative_close(actual: Real, expected: Real) {
        let tolerance = expected.abs().max(1.0) * 1.0e-14;
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual} expected={expected} diff={}",
            (actual - expected).abs()
        );
    }
}
