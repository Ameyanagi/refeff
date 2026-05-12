//! Debye and Einstein-model cumulant helpers ported from FEFF.
//!
//! This module starts with `DEBYE/sigm3.f90`, the correlated Einstein model
//! with a Morse potential used for first and third cumulant estimates.

use ndarray::ArrayView2;

use crate::{Real, atomic::atomic_weight as feff_atomic_weight};

const BOHR_ANGSTROM: Real = 0.529_177_249;
const HBAR: Real = 1.054_572_7e-34_f32 as Real;
const ATOMIC_MASS_UNIT: Real = 1.660_54e-27_f32 as Real;
const BOLTZMANN: Real = 1.380_658e-23_f32 as Real;
const DEBYE_CORRELATION_FACTOR: Real = 48.508_46_f32 as Real;
const DEBYE_ROMBERG_TOLERANCE: Real = 1.0e-5;
const DEBYE_ROMBERG_MAX_ITERATIONS: usize = 10;

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
    /// Path coordinates must be an `(nleg + 1) x 3` array.
    #[error(
        "Debye path coordinates must have at least 2 rows and exactly 3 columns, got {rows}x{columns}"
    )]
    InvalidPathShape { rows: usize, columns: usize },
    /// The atomic-number list must align with the coordinate rows.
    #[error("Debye path has {positions} coordinate rows but {atomic_numbers} atomic numbers")]
    InvalidAtomicNumberCount {
        positions: usize,
        atomic_numbers: usize,
    },
    /// Consecutive path coordinates must not be identical.
    #[error("Debye path leg {leg} has zero length")]
    ZeroLengthPathLeg { leg: usize },
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

/// Port of FEFF `sigms`: quantum Debye-Waller factor for a scattering path.
///
/// `positions_angstrom` is an `(nleg + 1) x 3` ndarray view. Row `0` and the
/// final row correspond to FEFF's central-atom endpoints for a closed path.
/// `atomic_numbers` must contain one atomic number for each row.
pub fn quantum_debye_waller_factor(
    temperature: Real,
    debye_temperature: Real,
    average_wigner_seitz_radius_bohr: Real,
    positions_angstrom: ArrayView2<'_, Real>,
    atomic_numbers: &[usize],
) -> Result<Real, DebyeError> {
    debye_waller_factor(DebyeWallerInput {
        temperature,
        debye_temperature,
        average_wigner_seitz_radius_bohr,
        positions_angstrom,
        atomic_numbers,
        correlation: quantum_debye_correlation,
    })
}

/// Port of FEFF `sigcl`: classical Debye-Waller factor for a scattering path.
pub fn classical_debye_waller_factor(
    temperature: Real,
    debye_temperature: Real,
    average_wigner_seitz_radius_bohr: Real,
    positions_angstrom: ArrayView2<'_, Real>,
    atomic_numbers: &[usize],
) -> Result<Real, DebyeError> {
    debye_waller_factor(DebyeWallerInput {
        temperature,
        debye_temperature,
        average_wigner_seitz_radius_bohr,
        positions_angstrom,
        atomic_numbers,
        correlation: classical_debye_correlation,
    })
}

type CorrelationFn =
    fn(Real, Real, Real, usize, usize, Real) -> Result<DebyeCorrelation, DebyeError>;

struct DebyeWallerInput<'a> {
    temperature: Real,
    debye_temperature: Real,
    average_wigner_seitz_radius_bohr: Real,
    positions_angstrom: ArrayView2<'a, Real>,
    atomic_numbers: &'a [usize],
    correlation: CorrelationFn,
}

fn debye_waller_factor(input: DebyeWallerInput<'_>) -> Result<Real, DebyeError> {
    let DebyeWallerInput {
        temperature,
        debye_temperature,
        average_wigner_seitz_radius_bohr,
        positions_angstrom,
        atomic_numbers,
        correlation,
    } = input;
    validate_path(positions_angstrom, atomic_numbers)?;
    ensure_nonnegative("tk", temperature)?;
    ensure_positive("thetad", debye_temperature)?;
    ensure_positive("rsavg", average_wigner_seitz_radius_bohr)?;

    let nleg = positions_angstrom.nrows() - 1;
    let mut total = 0.0;
    for il in 1..=nleg {
        for jl in il..=nleg {
            let cij = correlation_between_path_atoms(
                positions_angstrom,
                atomic_numbers,
                il,
                jl,
                DebyePathCorrelation {
                    debye_temperature,
                    temperature,
                    average_wigner_seitz_radius_bohr,
                    correlation,
                },
            )?;
            let cimjm = correlation_between_path_atoms(
                positions_angstrom,
                atomic_numbers,
                il - 1,
                jl - 1,
                DebyePathCorrelation {
                    debye_temperature,
                    temperature,
                    average_wigner_seitz_radius_bohr,
                    correlation,
                },
            )?;
            let cijm = correlation_between_path_atoms(
                positions_angstrom,
                atomic_numbers,
                il,
                jl - 1,
                DebyePathCorrelation {
                    debye_temperature,
                    temperature,
                    average_wigner_seitz_radius_bohr,
                    correlation,
                },
            )?;
            let cimj = correlation_between_path_atoms(
                positions_angstrom,
                atomic_numbers,
                il - 1,
                jl,
                DebyePathCorrelation {
                    debye_temperature,
                    temperature,
                    average_wigner_seitz_radius_bohr,
                    correlation,
                },
            )?;
            let first_leg = path_segment(positions_angstrom, il)?;
            let second_leg = path_segment(positions_angstrom, jl)?;
            let first_norm = vector_norm(first_leg);
            let second_norm = vector_norm(second_leg);
            let leg_projection = dot(first_leg, second_leg) / (first_norm * second_norm);
            let mut contribution = cij + cimjm - cijm - cimj;
            if jl != il {
                contribution *= 2.0;
            }
            total += contribution * leg_projection;
        }
    }

    let value = total / 4.0;
    ensure_finite_output("sig2", value)?;
    Ok(value)
}

#[derive(Clone, Copy)]
struct DebyePathCorrelation {
    debye_temperature: Real,
    temperature: Real,
    average_wigner_seitz_radius_bohr: Real,
    correlation: CorrelationFn,
}

fn validate_path(
    positions: ArrayView2<'_, Real>,
    atomic_numbers: &[usize],
) -> Result<(), DebyeError> {
    if positions.nrows() < 2 || positions.ncols() != 3 {
        return Err(DebyeError::InvalidPathShape {
            rows: positions.nrows(),
            columns: positions.ncols(),
        });
    }
    if positions.nrows() != atomic_numbers.len() {
        return Err(DebyeError::InvalidAtomicNumberCount {
            positions: positions.nrows(),
            atomic_numbers: atomic_numbers.len(),
        });
    }
    for value in positions.iter().copied() {
        ensure_finite("path coordinate", value)?;
    }
    for &atomic_number in atomic_numbers {
        atomic_weight(atomic_number)?;
    }
    Ok(())
}

fn correlation_between_path_atoms(
    positions: ArrayView2<'_, Real>,
    atomic_numbers: &[usize],
    first: usize,
    second: usize,
    path_correlation: DebyePathCorrelation,
) -> Result<Real, DebyeError> {
    let distance = path_distance(positions, first, second);
    Ok((path_correlation.correlation)(
        distance,
        path_correlation.debye_temperature,
        path_correlation.temperature,
        atomic_numbers[first],
        atomic_numbers[second],
        path_correlation.average_wigner_seitz_radius_bohr,
    )?
    .value)
}

fn path_distance(positions: ArrayView2<'_, Real>, first: usize, second: usize) -> Real {
    vector_norm([
        positions[(first, 0)] - positions[(second, 0)],
        positions[(first, 1)] - positions[(second, 1)],
        positions[(first, 2)] - positions[(second, 2)],
    ])
}

fn path_segment(positions: ArrayView2<'_, Real>, leg: usize) -> Result<[Real; 3], DebyeError> {
    let segment = [
        positions[(leg, 0)] - positions[(leg - 1, 0)],
        positions[(leg, 1)] - positions[(leg - 1, 1)],
        positions[(leg, 2)] - positions[(leg - 1, 2)],
    ];
    if vector_norm(segment) == 0.0 {
        Err(DebyeError::ZeroLengthPathLeg { leg })
    } else {
        Ok(segment)
    }
}

fn vector_norm(vector: [Real; 3]) -> Real {
    dot(vector, vector).sqrt()
}

fn dot(left: [Real; 3], right: [Real; 3]) -> Real {
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| left * right)
        .sum()
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
    feff_atomic_weight(atomic_number)
        .map_err(|_| DebyeError::InvalidAtomicNumber { z: atomic_number })
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

    #[test]
    fn debye_waller_factors_match_feff_reference() -> Result<(), DebyeError> {
        let copper_path = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.55, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let copper_atomic_numbers = [29, 29, 29];
        assert_close(
            quantum_debye_waller_factor(
                300.0,
                400.0,
                2.7,
                copper_path.view(),
                &copper_atomic_numbers,
            )?,
            5.620_717_932_013_852e-3,
        );
        assert_close(
            classical_debye_waller_factor(
                300.0,
                400.0,
                2.7,
                copper_path.view(),
                &copper_atomic_numbers,
            )?,
            5.216_382_857_388_334e-3,
        );

        let triangle_path = ndarray::arr2(&[
            [0.0, 0.0, 0.0],
            [1.91, 0.25, 0.10],
            [2.60, 1.40, -0.20],
            [0.0, 0.0, 0.0],
        ]);
        let triangle_atomic_numbers = [29, 8, 29, 29];
        assert_close(
            quantum_debye_waller_factor(
                180.0,
                650.0,
                2.3,
                triangle_path.view(),
                &triangle_atomic_numbers,
            )?,
            2.623_124_881_997_499_5e-3,
        );
        assert_close(
            classical_debye_waller_factor(
                180.0,
                650.0,
                2.3,
                triangle_path.view(),
                &triangle_atomic_numbers,
            )?,
            1.796_449_763_322_294e-3,
        );
        Ok(())
    }

    #[test]
    fn debye_waller_factors_reject_invalid_inputs() {
        let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        assert!(matches!(
            quantum_debye_waller_factor(300.0, 400.0, 2.7, positions.view(), &[29, 29]),
            Err(DebyeError::ZeroLengthPathLeg { leg: 1 })
        ));
        assert!(matches!(
            quantum_debye_waller_factor(300.0, 400.0, 2.7, positions.view(), &[29]),
            Err(DebyeError::InvalidAtomicNumberCount { .. })
        ));
        let bad_shape = ndarray::Array2::<Real>::zeros((1, 3));
        assert!(matches!(
            quantum_debye_waller_factor(300.0, 400.0, 2.7, bad_shape.view(), &[29]),
            Err(DebyeError::InvalidPathShape { .. })
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
