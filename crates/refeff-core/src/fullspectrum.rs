//! FEFF FULLSPECTRUM numerical helpers.
//!
//! This module covers small kernels from `FULLSPECTRUM/` that can be tested
//! independently of the full driver. Larger spectrum assembly remains in the
//! module runner layer until the surrounding FEFF state is ported.

use ndarray::{Array1, ArrayView1};
use thiserror::Error;

use crate::{Complex, Real};

/// FEFF Hartree energy in eV, matching `COMMON/m_constants.f90`.
pub const FEFF_HARTREE_EV: Real = 27.211_396;
/// FEFF Bohr radius in Angstrom, matching `COMMON/m_constants.f90`.
pub const FEFF_BOHR_ANGSTROM: Real = 0.529_177_249;
/// Inverse fine-structure constant used by FEFF optical sum rules.
pub const FEFF_ALPHA_INV: Real = 137.035_989_56;

/// Inputs for FEFF `FULLSPECTRUM/qsum.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumQSumInput<'a> {
    /// Number density `numden` used in the oscillator-strength sum rule.
    pub number_density: Real,
    /// Imaginary dielectric function `eps2`.
    pub epsilon2: ArrayView1<'a, Real>,
    /// Energy grid `omega`.
    pub omega: ArrayView1<'a, Real>,
    /// Number of active rows, equivalent to FEFF `iepts`.
    pub active_len: usize,
}

/// Inputs for FEFF `FULLSPECTRUM/sumrules.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumSumRulesInput<'a> {
    /// Number density `numden` in FEFF atomic units.
    pub number_density: Real,
    /// Photon energy grid in eV, as read from `opconsKK.dat`.
    pub energy_ev: ArrayView1<'a, Real>,
    /// Dielectric function minus one, using the columns written by `opcons.f90`.
    pub epsilon_minus_one: ArrayView1<'a, Complex>,
    /// Refractive index minus one, using the columns written by `opcons.f90`.
    pub refractive_index_minus_one: ArrayView1<'a, Complex>,
    /// FEFF `mu` absorption coefficient column, in `cm^(-1)`.
    pub absorption_coefficient: ArrayView1<'a, Real>,
}

/// Cumulative rows written to FEFF `sumrules.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct FullSpectrumSumRules {
    /// Photon energy grid in eV.
    pub energy_ev: Array1<Real>,
    /// Cumulative `epsilon_2` sum-rule effective electron count.
    pub epsilon2_effective_electrons: Array1<Real>,
    /// Cumulative absorption-coefficient sum-rule effective electron count.
    pub absorption_effective_electrons: Array1<Real>,
    /// Cumulative loss-function sum-rule effective electron count.
    pub loss_effective_electrons: Array1<Real>,
    /// Cumulative `mu * (n - 1)` sum-rule column.
    pub absorption_refractive_sum: Array1<Real>,
    /// Cumulative `(n - 1)` signed-to-absolute integral ratio.
    pub refractive_index_sum_ratio: Array1<Real>,
    /// Cumulative logarithmic loss-function moment ratio.
    pub log_loss_moment_ratio: Array1<Real>,
}

impl FullSpectrumSumRules {
    /// Number of cumulative sum-rule samples.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }
}

/// Error returned by FULLSPECTRUM helper kernels.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum FullSpectrumError {
    /// Number density must be positive.
    #[error("FULLSPECTRUM {name} must be positive, got {value}")]
    NonPositiveInput { name: &'static str, value: Real },
    /// Scalar inputs must be finite.
    #[error("FULLSPECTRUM {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    /// Active rows must fit in both input arrays.
    #[error("FULLSPECTRUM active row count {active_len} exceeds {field} length {len}")]
    ActiveCountOutOfRange {
        field: &'static str,
        active_len: usize,
        len: usize,
    },
    /// Array values must be finite.
    #[error("FULLSPECTRUM {field} row {row} must be finite, got {value}")]
    NonFiniteValue {
        field: &'static str,
        row: usize,
        value: Real,
    },
    /// Energy rows are expected in nondecreasing order for the trapezoid rule.
    #[error("FULLSPECTRUM omega row {row} must not decrease, got {current} after {previous}")]
    DecreasingOmega {
        row: usize,
        previous: Real,
        current: Real,
    },
    /// Tabulated sum-rule inputs require at least one row.
    #[error("FULLSPECTRUM {name} requires at least one row")]
    EmptyTable { name: &'static str },
    /// Array lengths must agree for a tabulated calculation.
    #[error("FULLSPECTRUM {field} length {actual} does not match energy length {expected}")]
    LengthMismatch {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    /// Each cumulative sum-rule value must be finite.
    #[error("FULLSPECTRUM sum-rule row {row} {field} must be finite, got {value}")]
    NonFiniteSumRule {
        field: &'static str,
        row: usize,
        value: Real,
    },
    /// The final sum-rule value must be finite.
    #[error("FULLSPECTRUM neff must be finite, got {value}")]
    NonFiniteResult { value: Real },
}

/// Port of `FULLSPECTRUM/qsum.f90`: compute the effective electron count.
///
/// FEFF applies a trapezoid integral to `omega * eps2`, then scales it by
/// `1 / (2*pi^2*numden)`. An active length of zero or one follows the Fortran
/// loop semantics and returns zero.
pub fn full_spectrum_effective_electron_count(
    input: FullSpectrumQSumInput<'_>,
) -> Result<Real, FullSpectrumError> {
    validate_positive("number_density", input.number_density)?;
    validate_active_len("epsilon2", input.active_len, input.epsilon2.len())?;
    validate_active_len("omega", input.active_len, input.omega.len())?;

    for row in 0..input.active_len {
        validate_finite_value("epsilon2", row, input.epsilon2[row])?;
        validate_finite_value("omega", row, input.omega[row])?;
        if row > 0 && input.omega[row] < input.omega[row - 1] {
            return Err(FullSpectrumError::DecreasingOmega {
                row,
                previous: input.omega[row - 1],
                current: input.omega[row],
            });
        }
    }

    let integral = (0..input.active_len.saturating_sub(1))
        .map(|row| {
            let left = input.omega[row] * input.epsilon2[row];
            let right = input.omega[row + 1] * input.epsilon2[row + 1];
            0.5 * (left + right) * (input.omega[row + 1] - input.omega[row])
        })
        .sum::<Real>();
    let result = integral / (2.0 * std::f64::consts::PI.powi(2) * input.number_density);
    if result.is_finite() {
        Ok(result)
    } else {
        Err(FullSpectrumError::NonFiniteResult { value: result })
    }
}

/// Port of `FULLSPECTRUM/sumrules.f90`: cumulative optical sum rules.
///
/// FEFF reads `opconsKK.dat`, recomputes the loss function from `epsilon - 1`,
/// converts energy and absorption units to atomic units, then writes a row after
/// every input point. This helper keeps the same integration rules, including
/// the right-endpoint rule for `mu` columns and trapezoids for energy-weighted
/// dielectric/loss columns.
pub fn full_spectrum_sum_rules(
    input: FullSpectrumSumRulesInput<'_>,
) -> Result<FullSpectrumSumRules, FullSpectrumError> {
    validate_positive("number_density", input.number_density)?;
    if input.energy_ev.is_empty() {
        return Err(FullSpectrumError::EmptyTable { name: "sum_rules" });
    }
    validate_matching_len(
        "epsilon_minus_one",
        input.epsilon_minus_one.len(),
        input.energy_ev.len(),
    )?;
    validate_matching_len(
        "refractive_index_minus_one",
        input.refractive_index_minus_one.len(),
        input.energy_ev.len(),
    )?;
    validate_matching_len(
        "absorption_coefficient",
        input.absorption_coefficient.len(),
        input.energy_ev.len(),
    )?;

    let mut epsilon2_effective_electrons = Vec::with_capacity(input.energy_ev.len());
    let mut absorption_effective_electrons = Vec::with_capacity(input.energy_ev.len());
    let mut loss_effective_electrons = Vec::with_capacity(input.energy_ev.len());
    let mut absorption_refractive_sum = Vec::with_capacity(input.energy_ev.len());
    let mut refractive_index_sum_ratio = Vec::with_capacity(input.energy_ev.len());
    let mut log_loss_moment_ratio = Vec::with_capacity(input.energy_ev.len());

    let scale = 1.0 / (2.0 * std::f64::consts::PI.powi(2) * input.number_density);
    let mut previous_energy_hartree = 0.0;
    let mut previous_epsilon2 = 0.0;
    let mut previous_loss = 0.0;
    let mut sum_epsilon2 = 0.0;
    let mut sum_absorption = 0.0;
    let mut sum_loss = 0.0;
    let mut sum_absorption_refractive = 0.0;
    let mut sum_refractive = 0.0;
    let mut sum_abs_refractive = 0.0;
    let mut sum_log_loss = 0.0;

    for row in 0..input.energy_ev.len() {
        let energy_ev = input.energy_ev[row];
        let epsilon = input.epsilon_minus_one[row];
        let refractive_index = input.refractive_index_minus_one[row];
        let absorption = input.absorption_coefficient[row];

        validate_finite_value("energy_ev", row, energy_ev)?;
        validate_finite_value("epsilon_minus_one real", row, epsilon.re)?;
        validate_finite_value("epsilon_minus_one imaginary", row, epsilon.im)?;
        validate_finite_value("refractive_index_minus_one real", row, refractive_index.re)?;
        validate_finite_value(
            "refractive_index_minus_one imaginary",
            row,
            refractive_index.im,
        )?;
        validate_finite_value("absorption_coefficient", row, absorption)?;
        if row > 0 && energy_ev < input.energy_ev[row - 1] {
            return Err(FullSpectrumError::DecreasingOmega {
                row,
                previous: input.energy_ev[row - 1],
                current: energy_ev,
            });
        }

        let energy_hartree = energy_ev / FEFF_HARTREE_EV;
        let mu_atomic = absorption * FEFF_BOHR_ANGSTROM / 1000.0;
        let epsilon2 = epsilon.im;
        let refractive_real = refractive_index.re;
        let loss = epsilon2 / (epsilon2.powi(2) + (epsilon.re + 1.0).powi(2));
        let delta_energy = energy_hartree - previous_energy_hartree;

        sum_epsilon2 += 0.5
            * (energy_hartree * epsilon2 + previous_energy_hartree * previous_epsilon2)
            * delta_energy;
        sum_absorption += mu_atomic * delta_energy;
        sum_loss +=
            0.5 * (energy_hartree * loss + previous_energy_hartree * previous_loss) * delta_energy;
        sum_absorption_refractive += mu_atomic * refractive_real * delta_energy;
        sum_refractive += delta_energy * refractive_real;
        sum_abs_refractive += delta_energy * refractive_real.abs();
        if previous_energy_hartree > 0.0 {
            sum_log_loss += 0.5
                * (energy_hartree.ln() * energy_hartree * loss
                    + previous_energy_hartree.ln() * previous_energy_hartree * previous_loss)
                * delta_energy;
        } else {
            sum_log_loss += energy_hartree.ln() * energy_hartree * loss * energy_hartree;
        }

        push_sum_rule_value(
            &mut epsilon2_effective_electrons,
            "epsilon2_effective_electrons",
            row,
            scale * sum_epsilon2,
        )?;
        push_sum_rule_value(
            &mut absorption_effective_electrons,
            "absorption_effective_electrons",
            row,
            FEFF_ALPHA_INV * scale * sum_absorption,
        )?;
        push_sum_rule_value(
            &mut loss_effective_electrons,
            "loss_effective_electrons",
            row,
            scale * sum_loss,
        )?;
        push_sum_rule_value(
            &mut absorption_refractive_sum,
            "absorption_refractive_sum",
            row,
            FEFF_ALPHA_INV * scale * sum_absorption_refractive,
        )?;
        push_sum_rule_value(
            &mut refractive_index_sum_ratio,
            "refractive_index_sum_ratio",
            row,
            sum_refractive / sum_abs_refractive,
        )?;
        push_sum_rule_value(
            &mut log_loss_moment_ratio,
            "log_loss_moment_ratio",
            row,
            sum_log_loss / sum_loss,
        )?;

        previous_energy_hartree = energy_hartree;
        previous_epsilon2 = epsilon2;
        previous_loss = loss;
    }

    Ok(FullSpectrumSumRules {
        energy_ev: input.energy_ev.to_owned(),
        epsilon2_effective_electrons: Array1::from_vec(epsilon2_effective_electrons),
        absorption_effective_electrons: Array1::from_vec(absorption_effective_electrons),
        loss_effective_electrons: Array1::from_vec(loss_effective_electrons),
        absorption_refractive_sum: Array1::from_vec(absorption_refractive_sum),
        refractive_index_sum_ratio: Array1::from_vec(refractive_index_sum_ratio),
        log_loss_moment_ratio: Array1::from_vec(log_loss_moment_ratio),
    })
}

fn validate_positive(name: &'static str, value: Real) -> Result<(), FullSpectrumError> {
    if !value.is_finite() {
        Err(FullSpectrumError::NonFiniteInput { name, value })
    } else if value <= 0.0 {
        Err(FullSpectrumError::NonPositiveInput { name, value })
    } else {
        Ok(())
    }
}

fn validate_matching_len(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), FullSpectrumError> {
    if actual == expected {
        Ok(())
    } else {
        Err(FullSpectrumError::LengthMismatch {
            field,
            actual,
            expected,
        })
    }
}

fn validate_active_len(
    field: &'static str,
    active_len: usize,
    len: usize,
) -> Result<(), FullSpectrumError> {
    if active_len > len {
        Err(FullSpectrumError::ActiveCountOutOfRange {
            field,
            active_len,
            len,
        })
    } else {
        Ok(())
    }
}

fn validate_finite_value(
    field: &'static str,
    row: usize,
    value: Real,
) -> Result<(), FullSpectrumError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FullSpectrumError::NonFiniteValue { field, row, value })
    }
}

fn push_sum_rule_value(
    values: &mut Vec<Real>,
    field: &'static str,
    row: usize,
    value: Real,
) -> Result<(), FullSpectrumError> {
    if value.is_finite() {
        values.push(value);
        Ok(())
    } else {
        Err(FullSpectrumError::NonFiniteSumRule { field, row, value })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};
    use num_complex::Complex64;

    use crate::Real;

    use super::{
        FullSpectrumError, FullSpectrumQSumInput, FullSpectrumSumRulesInput,
        full_spectrum_effective_electron_count, full_spectrum_sum_rules,
    };

    #[test]
    fn effective_electron_count_matches_feff_qsum_reference() -> Result<(), FullSpectrumError> {
        let omega = array![0.0, 0.1, 0.2, 0.5, 1.0, 1.8];
        let epsilon2 = array![0.0, 0.5, 1.0, 0.25, 0.75, 0.1];

        let neff = full_spectrum_effective_electron_count(FullSpectrumQSumInput {
            number_density: 0.075,
            epsilon2: epsilon2.view(),
            omega: omega.view(),
            active_len: 6,
        })?;

        assert_close(neff, 0.442_098_097_959_400_5, 1.0e-14);
        Ok(())
    }

    #[test]
    fn effective_electron_count_matches_feff_single_point_reference()
    -> Result<(), FullSpectrumError> {
        let omega = array![0.0, 0.1];
        let epsilon2 = array![1.0, 2.0];

        let neff = full_spectrum_effective_electron_count(FullSpectrumQSumInput {
            number_density: 0.075,
            epsilon2: epsilon2.view(),
            omega: omega.view(),
            active_len: 1,
        })?;

        assert_eq!(neff, 0.0);
        Ok(())
    }

    #[test]
    fn effective_electron_count_rejects_invalid_inputs() {
        let omega = array![0.0, 0.1, 0.2];
        let epsilon2 = array![0.0, 0.5, 1.0];

        assert!(matches!(
            full_spectrum_effective_electron_count(FullSpectrumQSumInput {
                number_density: 0.0,
                epsilon2: epsilon2.view(),
                omega: omega.view(),
                active_len: 3,
            }),
            Err(FullSpectrumError::NonPositiveInput {
                name: "number_density",
                ..
            })
        ));
        assert!(matches!(
            full_spectrum_effective_electron_count(FullSpectrumQSumInput {
                number_density: 0.075,
                epsilon2: epsilon2.view(),
                omega: omega.view(),
                active_len: 4,
            }),
            Err(FullSpectrumError::ActiveCountOutOfRange {
                field: "epsilon2",
                ..
            })
        ));
        assert!(matches!(
            full_spectrum_effective_electron_count(FullSpectrumQSumInput {
                number_density: 0.075,
                epsilon2: array![0.0, f64::NAN, 1.0].view(),
                omega: omega.view(),
                active_len: 3,
            }),
            Err(FullSpectrumError::NonFiniteValue {
                field: "epsilon2",
                row: 1,
                ..
            })
        ));
        assert!(matches!(
            full_spectrum_effective_electron_count(FullSpectrumQSumInput {
                number_density: 0.075,
                epsilon2: epsilon2.view(),
                omega: array![0.0, 0.2, 0.1].view(),
                active_len: 3,
            }),
            Err(FullSpectrumError::DecreasingOmega { row: 2, .. })
        ));
    }

    #[test]
    fn sum_rules_match_feff_reference_algorithm() -> Result<(), FullSpectrumError> {
        let energy_ev = array![10.0, 20.0, 40.0];
        let epsilon_minus_one = array![
            Complex64::new(0.10, 0.20),
            Complex64::new(0.15, 0.25),
            Complex64::new(0.20, 0.35),
        ];
        let refractive_index_minus_one = array![
            Complex64::new(0.01, 0.02),
            Complex64::new(0.02, 0.03),
            Complex64::new(0.03, 0.04),
        ];
        let absorption_coefficient = array![1000.0, 2000.0, 3000.0];

        let rules = full_spectrum_sum_rules(FullSpectrumSumRulesInput {
            number_density: 0.075,
            energy_ev: energy_ev.view(),
            epsilon_minus_one: epsilon_minus_one.view(),
            refractive_index_minus_one: refractive_index_minus_one.view(),
            absorption_coefficient: absorption_coefficient.view(),
        })?;

        assert_eq!(rules.point_count(), 3);
        assert_close(rules.energy_ev[2], 40.0, 0.0);
        assert_close(
            rules.epsilon2_effective_electrons[2],
            0.214_375_530_814_624_3,
            1.0e-14,
        );
        assert_close(
            rules.absorption_effective_electrons[2],
            162.008_009_804_073_2,
            1.0e-14,
        );
        assert_close(
            rules.loss_effective_electrons[2],
            0.145_731_231_111_208_3,
            1.0e-14,
        );
        assert_close(
            rules.absorption_refractive_sum[2],
            4.140_204_694_992_98,
            1.0e-14,
        );
        assert_close(rules.refractive_index_sum_ratio[2], 1.0, 1.0e-14);
        assert_close(
            rules.log_loss_moment_ratio[2],
            -0.038_690_505_695_948_56,
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn sum_rules_reject_invalid_inputs() {
        let energy_ev = array![10.0, 5.0];
        let epsilon_minus_one = array![Complex64::new(0.10, 0.20), Complex64::new(0.15, 0.25)];
        let refractive_index_minus_one =
            array![Complex64::new(0.01, 0.02), Complex64::new(0.02, 0.03)];
        let absorption_coefficient = array![1000.0, 2000.0];

        assert!(matches!(
            full_spectrum_sum_rules(FullSpectrumSumRulesInput {
                number_density: 0.075,
                energy_ev: energy_ev.view(),
                epsilon_minus_one: epsilon_minus_one.view(),
                refractive_index_minus_one: refractive_index_minus_one.view(),
                absorption_coefficient: absorption_coefficient.view(),
            }),
            Err(FullSpectrumError::DecreasingOmega { row: 1, .. })
        ));
        assert!(matches!(
            full_spectrum_sum_rules(FullSpectrumSumRulesInput {
                number_density: 0.075,
                energy_ev: Array1::<Real>::zeros(0).view(),
                epsilon_minus_one: Array1::<Complex64>::zeros(0).view(),
                refractive_index_minus_one: Array1::<Complex64>::zeros(0).view(),
                absorption_coefficient: Array1::<Real>::zeros(0).view(),
            }),
            Err(FullSpectrumError::EmptyTable { name: "sum_rules" })
        ));
        assert!(matches!(
            full_spectrum_sum_rules(FullSpectrumSumRulesInput {
                number_density: 0.075,
                energy_ev: array![10.0, 20.0].view(),
                epsilon_minus_one: array![Complex64::new(0.10, 0.20)].view(),
                refractive_index_minus_one: refractive_index_minus_one.view(),
                absorption_coefficient: absorption_coefficient.view(),
            }),
            Err(FullSpectrumError::LengthMismatch {
                field: "epsilon_minus_one",
                ..
            })
        ));
        assert!(matches!(
            full_spectrum_sum_rules(FullSpectrumSumRulesInput {
                number_density: 0.075,
                energy_ev: array![10.0].view(),
                epsilon_minus_one: array![Complex64::new(0.10, 0.20)].view(),
                refractive_index_minus_one: array![Complex64::new(0.0, 0.0)].view(),
                absorption_coefficient: array![1000.0].view(),
            }),
            Err(FullSpectrumError::NonFiniteSumRule {
                field: "refractive_index_sum_ratio",
                ..
            })
        ));
    }

    fn assert_close(actual: Real, expected: Real, tolerance: Real) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }
}
